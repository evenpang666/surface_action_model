# SAM 3 + PI0.5：四视角动作与双视角 BB

当前配对 OBB 模型与训练/标注入口见 [独立 OA-VLA](../oa-vla/README.md)，配置 format 2。
本文下述 schema-5 集成版是历史独立水平框实现，不包含跨视角对象配对；旧权重须迁移并再训练。

当前默认模型为 `Sam3VlaModelPytorch`，checkpoint schema 5。它直接继承 PI0.5，不构建点云
prefix、点云 DiT 或点云输出头；训练和推理也不需要 `robot_points`、`current_surface` 或点流标签。
历史点云模型仅由显式 legacy 入口加载。

## 连接位置和相机隔离

```text
front / side / right_wrist / left_wrist RGB + 指令/状态
    → 共享 SigLIP/PaliGemma → 四视角 prefix KV → action expert → flow matching → 动作块

front / side RGB + 同一指令/状态（单独构建，不读取四视角 prefix）
    → 同一套 SigLIP/PaliGemma → 图文 prefix hidden states
    → 图像/文本维度适配层 + 二维位置编码 + camera embedding
    → SAM 3 object-query decoder → front、side 的 BB / score / presence
```

检测分支的接入点是 PaliGemma prefix 的输出。额外运行一个仅含 front/side/text 的 prefix，
因为从四视角融合后的输出截取 front/side token 仍会携带腕部图像信息。
独立 prefix 增加了两视角的编码/融合计算，但不增加第二套视觉骨干权重。
检测只在每次观测计算一次，与动作 flow matching 的迭代次数无关。

检测解码器来自 [Meta 官方 SAM 3](https://github.com/facebookresearch/sam3)，固定源码提交
`7f473cfd983d5eb43040ff608f66bff0d7bdcb53`。保留 query、文本交叉注意力、box RPB、迭代框细化、
presence token/head 与图文匹配评分。省去 PE 骨干、mask、视频 tracker；用 PaliGemma 的图文融合代替原融合编码器。
原生模块和许可见 [NOTICE](../openpi/src/openpi/models_pytorch/sam3_detector/NOTICE.md)。
默认 256 维、6 层、200 queries，与官方检测解码器形状一致；本项目使用 one-to-one Hungarian 匹配，
不启用原训练的 DAC one-to-many 分支。

这保留了 SAM 3 的检测模块，但不能视为完整 SAM 3 检测能力的无损迁移：更换了输入特征分布，适配层需要训练。

## 数据

```bash
python scripts/preprocess_quest3_rgb_action.py \
  --episode /path/to/episode.hdf5 --output outputs/rgb_action/episode_00000.shard
```

四视角的 RGB 路径为 `observations/images/{front,side,right_wrist,left_wrist}/rgb`，
图像与 qpos、动作和时间戳共享帧轴。支持不同相机分辨率；每路分别 letterbox 到 224。
缺少或被 mask 的相机直接报错，不会隐式零填充。新预处理无需深度、网格 FK 或点云跟踪。

既有四视角 action shard 可以继续使用；点云字段不会加载。单 front 或缺少腕部视角的数据需要重新导出。
`sample_frame_indices` 将 frame-aligned RGB/qpos 与 sample-aligned action chunk 对齐。
动作 horizon 边界使用 step mask，padding 动作不参与损失或归一化统计。

检测标签支持两种来源：

- 源 HDF5 的 `annotations/obstacle_obbs`（`dual_rgb_obb_annotation_v2`）：只使用已人工审核帧，
  将每个旋转框转换为原图中的外接轴对齐框、裁到图像范围，再映射到 padded 224 图像。
- shard 内的 `detection_boxes[N,2,M,4]`、`detection_valid[N,2,M]`、`detection_reviewed[N,2]`。
  同时声明 `detection_views=['front','side']` 与
  `detection_box_format='cxcywh_normalized_padded'`。标签与 action sample 轴对齐。

每视角最多 32 个 GT 框，批处理自动 padding。未审核视角不计算检测损失；已审核空场景训练 presence=0。
沿用的障碍框标签表示全部标注障碍物，不会自动变成“当前操作对象/放置目标”的角色标签。

## 训练与权重

```bash
python scripts/train_pi05_sam3_pytorch.py \
  --dataset outputs/rgb_action --output outputs/models/sam3_vla.pt --validate-only
python scripts/train_pi05_sam3_pytorch.py \
  --dataset outputs/rgb_action --output outputs/models/sam3_vla.pt \
  --sam3-checkpoint /path/to/sam3.pt
```

PI0.5 权重默认取项目 YAML 的 `model.pretrained`。`--sam3-checkpoint` 读取官方 checkpoint 中
`detector.transformer.decoder.*` 和 `detector.dot_prod_scoring.*`（也支持已去掉 `detector.` 的键），
严格检查两个模块是否完整；其余视觉/分割/跟踪权重不加载。特征适配层与 camera embedding 随机初始化。
省略该参数会明确提示检测解码器随机初始化，不会自动下载或声称加载了预训练 SAM 3 权重。

默认联合损失为 action flow matching + detection loss；检测包含 Hungarian 匹配后的 L1/GIoU、
局部匹配 BCE、全局 presence BCE，以及中间 decoder 层辅助损失。最终置信度为局部分数 × presence。
检测使用无随机几何增强的 prefix，保持框和图像对齐；动作分支保留原 PI0.5 增强。

| 参数 | 用法 |
| --- | --- |
| `--detection-stage joint` | 默认，训练动作 expert 与检测头，冻结 VLM |
| `--detection-stage action` | 仅动作训练；四视角仍必须齐全，无需检测标注 |
| `--detection-stage detection` | 仅检测头训练，保留动作权重 |
| `--detection-loss-weight 1` | 检测损失权重 |
| `--sam3-checkpoint PATH` | 初始化官方检测 decoder/scorer，默认形状才能完整加载 |
| `--detection-queries 200` / `--detection-layers 6` | 原版形状；调整后通常需从头训练或另备对应 checkpoint |
| `--resume PATH` | schema-5 模型；同阶段恢复 optimizer，切换阶段只恢复模型 |
| `--init-from PATH` | 加载旧动作/点云 checkpoint 的 PI0.5 权重，丢弃旧 point/detection 模块 |
| `--no-freeze-base` | 同时训练 VLM，需要更大显存；检测隔离仍然成立 |

分阶段例子：

```bash
python scripts/train_pi05_sam3_pytorch.py \
  --dataset outputs/rgb_action --output outputs/models/action.pt --detection-stage action
python scripts/train_pi05_sam3_pytorch.py \
  --dataset outputs/rgb_action --output outputs/models/sam3_vla.pt \
  --resume outputs/models/action.pt --detection-stage detection
```

若希望阶段切换时再初始化官方检测权重，先用 `--sam3-checkpoint` 完成第一阶段的模型初始化；
`--resume` 不允许重新覆盖头权重。旧模型迁移使用 `--init-from`，同时可指定 `--sam3-checkpoint`。
旧点云条件被移除且相机数量改变，迁移后仍须重新训练和评估动作。

## 策略服务和输出

```bash
python scripts/serve_quest3_pi05_safety.py --checkpoint outputs/models/sam3_vla.pt
```

请求：

```python
{
    'prompt': 'move the instrument around the obstacle',
    'qpos': np.zeros(12, dtype=np.float32),
    'rgb_front': front_rgb,          # uint8 [H,W,3]
    'rgb_side': side_rgb,
    'rgb_right_wrist': right_rgb,
    'rgb_left_wrist': left_rgb,
}
```

响应 schema 5：

```python
{
    'actions': ...,                 # [H,14]，单臂训练为 [H,7]
    'action_chunk': ...,            # 同上
    'detections': {
        'front': [{'xyxy': [x0,y0,x1,y1], 'score': ...}, ...],
        'side':  [{'xyxy': [x0,y0,x1,y1], 'score': ...}, ...],
    },
    'detection_box_format': 'xyxy_original_pixels',
    'detection_status': 'ok',
}
```

各视角使用各自原图尺寸，还原 letterbox 并裁边。无 wrist 检测字段，无点云字段。
`--score-threshold` 默认 0.5。仅 action 训练的 checkpoint 返回空框及 `detection_status=not_trained`，
不发布未训练头的随机框。

可用四张本地图像调用服务，不连接机器人：

```bash
python scripts/query_pi05_sam3_policy.py --prompt 'avoid obstacle' \
  --front front.png --side side.png --right-wrist right.png --left-wrist left.png \
  --qpos 0 0 0 0 0 0 0 0 0 0 0 0
```

旧 `run_ur7e_vla_safety_executor.py` 依赖点流及其 CBF 协议，不适配 schema 5；二维 BB 不直接替代三维避障约束。
需要继续运行旧 checkpoint 时使用 `scripts/serve_quest3_pi05_surface_legacy.py`。

## 验证

```bash
OMP_NUM_THREADS=1 python -m pytest -q verification/test_sam3_vla.py
```

包含腕部输入扰动下检测输出严格不变、动作对腕部输入敏感、联合/检测阶段梯度、未审核/空帧监督、
框坐标还原、原生权重键加载、四路输入检查、无点云 shard、checkpoint 服务重载及预处理→真实单步训练入口。
测试使用小型模型，不代表真实机器人性能验证。
