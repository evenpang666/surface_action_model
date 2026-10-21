# oavla

独立的 **oavla** 模型：四视角 RGB 驱动动作 flow matching，SAM 3 检测分支只使用 front/side，
直接输出当前帧的**对象对**：每个对象包含 front/side 的旋转 OBB、可见性和配对编号，无需语义类别名称。无点云输入或输出头。

本目录可整体复制到其他机器后单独安装。运行时不依赖 OpenPI、JAX、Flax、父目录的
`safety_module`、机器人 SDK 或其配置。定制的 Gemma/SigLIP/PaliGemma 是包内私有实现，
无需覆盖 `site-packages` 或替换 Transformers 的全局模型类。

项目展示名称为 **oavla**；目录与安装包名为 `oa-vla`，命令以 `oa-vla-` 开头，Python 导入名为 `oa_vla`。

[安装](#安装) · [模型与相机](#模型与相机) · [Python API](#python-api) · [标注](#独立数据标注) · [训练](#训练) · [socket 服务](#tcp-socket-服务与发送脚本)

## 安装

Python 3.11 或 3.12，建议新建环境。在本目录执行：

```bash
python -m pip install -e '.[data,test]'
```

仅使用模型与推理可 `python -m pip install .`。`data` 额外依赖 h5py，用于读取已有 HDF5 框标注。
PyTorch 2.7.1 / torchvision 0.22.1 / Transformers 4.53.2 已在 pyproject 中声明。
模型权重和 SentencePiece tokenizer 不随代码分发。

## 结构

```text
oa-vla/
├── pyproject.toml
├── configs/                 # default.json / tiny.json
├── examples/                # 合成数据与 CPU 全流程 smoke test
├── tests/                   # 包内测试，含父项目依赖隔离
└── src/oa_vla/
    ├── config.py            # 独立配置
    ├── model.py             # OAVLA + SAM 3 decoder + 对象 query 跨视角融合
    ├── geometry.py          # 旋转框、可微重叠代理及原图坐标恢复
    ├── annotate.py          # 独立 HDF5 双视角 OBB 标注工具
    ├── flow.py              # 连续动作 flow matching
    ├── backbone.py          # PaliGemma / action expert 联合注意力
    ├── observation.py       # tensor 输入
    ├── preprocessing.py     # 纯 PyTorch 图像预处理
    ├── tokenizer.py         # SentencePiece / 测试专用 byte tokenizer
    ├── data.py              # 四视角 RGB/action shard
    ├── checkpoint.py        # config + safetensors 独立导出
    ├── policy.py            # 高层 RGB/state/prompt API
    ├── train.py             # 独立训练入口
    ├── predict.py           # 本地图像推理入口
    ├── serve.py             # TCP socket 模型服务
    ├── send.py              # 四路 RGB 文件发送客户端
    ├── socket_protocol.py   # 长度前缀 MessagePack 协议与帧校验
    ├── init_test_checkpoint.py  # tiny 随机 checkpoint 生成器
    ├── convert.py           # 旧 format-1 / schema-5 权重迁移（须再训练）
    └── vendor/              # 有来源及许可的 SAM 3 / Transformers 模块
```

## 模型与相机

```text
front / side / right_wrist / left_wrist + text/state
    → SigLIP + PaliGemma → action expert → action chunk

front / side + text/state（独立 prefix，复用相同骨干权重）
    → PaliGemma hidden states → 特征适配层
    → SAM 3 decoder + 每层双视角对象 query 融合
    → 对象存在分数 + 两视角 OBB / 可见性
    → 按对象对执行一次 Hungarian 匹配
```

检测使用独立 front/side/text prefix，既不读取腕部图像，也不读取已经融合了腕部图像的文本 token。
四路 RGB 缺一即报错。动作可受两路腕部图像影响，检测输出在单独改变任一腕部输入时保持严格不变。
两个 prefix 增加计算量，但不增加第二套视觉骨干权重。

默认图像分辨率为 letterbox 后的 224×224。不同相机可提供不同原始尺寸。
每个对象 query 保留 front/side 独立的空间参考框，在每层解码后交换两个视角的 query 特征。
同一 query 的两个框由同一个标注对象监督；只共享 query 初始权重、分别匹配两路框无法实现这一点。
对象编号仅表示配对，不表示类别或操作角色。

## Python API

```python
import numpy as np
from oa_vla import OAVLA, OAVLAConfig, OAVLAPolicy, Observation

# 加载包含模型、归一化参数和 tokenizer 的独立导出目录
policy = OAVLAPolicy.from_pretrained('/models/oa-vla', device='cuda')
result = policy.predict(
    rgb={
        'front': front_rgb,              # uint8 [H,W,3]
        'side': side_rgb,
        'right_wrist': right_wrist_rgb,
        'left_wrist': left_wrist_rgb,
    },
    state=np.asarray(qpos, dtype=np.float32),  # 默认 12-D，使用训练时相同的状态定义
    prompt='move the instrument around the obstacle',
    num_steps=10,
)
# result['actions']: [action_horizon, logical_action_dim]，默认 14-D
# result['objects']: [{'pair_id':0, 'score':..., 'front': {...}, 'side': {...}}, ...]
# 每侧: {'obb':[cx,cy,w,h,angle_deg], 'corners':[[x,y],...], 'visibility_score':..., 'score':...}
# result['detections']['front'/'side']: 同一结果按视角展开，仍包含 pair_id
```

OBB 是各视角原始 RGB 像素坐标，角度为图像坐标中的顺时针度数，长边为 width；无 wrist 检测结果。
不可见侧为 `null`；两个视角均未通过可见性阈值时不发布该对象。`score_threshold` 控制对象存在分数，
`visibility_threshold` 控制每侧可见性（局部分数 × 该视角全局 presence），均默认 0.5。
`pair_id` 为当前预测的 query 编号，只在本次输出中配对，不保证跨帧稳定，也不保证等于人工标注的名称。
同类、多对象均不需要类别名称。角点不裁剪到图像边界，以保留 OBB 几何。
低层 `sample_actions_and_boxes` 返回 actions、`obbs[B,2,Q,6]`、`pair_scores[B,Q]`、
`visibility_scores[B,2,Q]`、`scores[B,2,Q]`，以及供原生 SAM 空间参考使用的 `boxes[B,2,Q,4]` 和 presence。
`Observation.images` 使用相同的物理视角名称，值为 `[-1,1]` float32 `[B,3,H,W]`，不使用旧内部相机键。

`OAVLA(config)` 创建模型；`model.save_pretrained(directory)` 与 `OAVLA.from_pretrained(directory)`
保存/加载纯模型。`OAVLAPolicy.save_pretrained(directory)` 另外保存归一化参数、处理器配置与 tokenizer。
未声明经过检测训练的 policy 返回空框和 `detection_status=not_trained`，不发布随机头的预测框。

## 独立数据标注

```bash
oa-vla-annotate --dataset /data/recordings --port 8765
# 或 python -m oa_vla.annotate --dataset /data/episode.hdf5
```

浏览器打开 `http://127.0.0.1:8765`。输入对象编号（例如 a、b），在 front 和 side 中为同一对象
使用相同编号，各画一个框并旋转；每个视角可标多个不同对象。编号只用于建立同帧对应关系。
同一视角重复编号会拒绝保存。对象在某侧不可见时，输入其编号，点击该侧“此编号不可见”；
至少一侧必须存在实际框。没有框且未显式标记不可见的对象不能保存，避免把漏标当作遮挡。
“标记无障碍”与尚未审核分开。保存直接写入原始 HDF5 的 `annotations/obstacle_obbs`，
保留 RGB、动作和时间戳；支持断点继续和 revision 冲突检测。

新标注 schema 为 `dual_rgb_paired_obb_v3`，每帧包含 `boxes.front/side` 和
`visibility[object_id][front/side]`。原 `dual_rgb_obb_annotation_v2` 标注可直接读取，
读取时按 object_id 对齐；所有对象必须在两视角各出现一次。旧重复编号应重新分配不同对象编号，
不能根据列表顺序猜配对。升级到 v3 后保留之前的 v2 帧记录。

原始 HDF5 需要 `timestamps_ns[T]` 及 `observations/images/{front,side}/rgb[T,H,W,3]`
（也支持逐帧 JPEG 字节）；独立标注工具不需要深度、机器人 SDK 或 OpenCV。
父项目的批量/相机标注界面也已支持相同配对及可见性字段。

## 训练

输入为独立 `.npz` 或内含 `.npy` 的 `.shard` 目录，必需字段：

| 字段 | 形状 / 说明 |
| --- | --- |
| `task_text` | 标量字符串 |
| `qpos` | `[T,state_dim]` |
| `rgb_front / rgb_side / rgb_right_wrist / rgb_left_wrist` | uint8 `[T,H,W,3]`，四路与 qpos 共享帧轴 |
| `sample_frame_indices` | `[N]`，每个样本的当前观测帧 |
| `action_chunks` | `[N,H,logical_action_dim]`，与 sample 轴对齐 |
| `action_step_mask` | 可选 `[N,H]`，padding 不参与损失和动作统计 |
| `detection_boxes` | `[N,2,M,6]`，`cx,cy,w,h,cos(2θ),sin(2θ)`，front、side 顺序 |
| `detection_object_ids` | `[N,2,M]` 字符串或整数；有效框必须提供，同对象跨视角相同 ID |
| `detection_valid / detection_reviewed` | `[N,2,M]` / `[N,2]` |
| `detection_views` | `['front','side']` |
| `detection_box_format` | `'cxcywh_cos2sin2_normalized_padded'` |

每帧最多 32 个对象（两视角 ID 的并集），每视角每对象最多一个框。不同视角的 M 槽位无需预先对应，
读取时用 ID 重新对齐到共同的对象槽位。中心使用 224 letterbox 坐标归一化，宽高除以原图最长边，
保持各向同性；角度保留原图方向。整数 resize 的亚像素舍入只影响中心映射，恢复原图 OBB 时使用相同约定。
宽高为正、角度向量为单位向量，loader 统一长短边及角度。无效槽位不参与框回归。
`reviewed=True` 表示该视角已完整审核；缺少某个对象的 valid 框表示确认不可见。
`reviewed=False` 表示未知，不训练该视角的几何或缺失监督。

没有内嵌标签时，支持 `source_episode` / `source_episode_relative` 引用的 HDF5 v2/v3 标注，
同时提供完整 `timestamps_ns[T]`；加载时按时间戳校验与 `sample_frame_indices` 对齐。
整目录迁移时需携带源 HDF5；内嵌标签可消除此数据依赖。没有 ID/角度的旧水平框标签会明确报错，
需从原始 OBB 标注重新读取，无法无损恢复丢失的信息。

训练只执行一次对象对 Hungarian 匹配，代价累加两个已审核视角的 OBB 几何、方向、旋转重叠代理及可见性。
损失包括对象存在 BCE、每侧可见性 BCE、每侧全局 presence BCE、OBB L1/方向损失，
以及由矩形协方差得到的 Bhattacharyya 重叠损失（可微旋转重叠代理，不是精确多边形 IoU）。
原生 SAM 水平参考框另用 OBB 外接框监督，以保留 box RPB。辅助 decoder 层采用相同的成对训练。
已审核空帧训练所有对象不存在；未审核视角不参与损失。

```bash
oa-vla-train --dataset /data/rgb_action --output /models/oa-vla --validate-only

oa-vla-train --dataset /data/rgb_action --output /models/oa-vla \
  --pretrained /weights/pi05_base/model.safetensors \
  --sam3-checkpoint /weights/sam3.pt \
  --tokenizer-model /weights/tokenizer.model \
  --device cuda --precision bfloat16
```

未指定 `--config` 时从数据解析 horizon、状态与逻辑动作维度；其他架构参数取默认配置。
指定 JSON 配置后数据必须匹配该配置。默认冻结视觉语言骨干，联合训练 action expert 和检测分支。
`--stage action` 或 `--stage detection` 支持分阶段训练；`--resume /models/oa-vla` 从独立目录继续训练。
相同训练阶段恢复 optimizer、GradScaler 与 epoch；切换阶段只保留模型和归一化参数。
多卡入口为 `torchrun --nproc-per-node N -m oa_vla.train ...`。

基础权重直接读取 PyTorch safetensors / state dict，不直接加载 JAX/Orbax checkpoint。
`--sam3-checkpoint` 仅加载原生检测 decoder/scorer，适配层、跨视角融合及旋转/对象头仍需要训练；省略它时会明确提示检测模块随机初始化。
不提供基础权重时需要显式 `--allow-random-init`，主要用于测试。

导出目录：

```text
/models/oa-vla/
├── config.json
├── model.safetensors
├── metadata.json            # 归一化、检测训练状态等
├── processor.json
├── tokenizer.model          # SentencePiece；tiny 测试使用 byte tokenizer，无此文件
└── training_state.pt        # optimizer / GradScaler / epoch，仅继续训练需要
```

模型配置与输出协议升级为 format/schema 2。新增配对与旋转参数后，旧模型需通过迁移命令初始化，
无法直接作为已训练的配对模型继续推理。导出目录自身不引用 OpenPI 源码路径。

## 转换之前的模型

```bash
# 上一版独立模型目录
oa-vla-convert --checkpoint /models/oa-vla-v1 --output /models/oa-vla-v2-init
# 更早的集成版 checkpoint
oa-vla-convert --checkpoint /models/sam3_vla.pt --output /models/oa-vla \
  --tokenizer-model /weights/tokenizer.model
```

接受独立 oavla format 1 导出目录，或 `Sam3VlaModelPytorch` schema 5 action/box checkpoint；不会导入 OpenPI 类。
保留已有骨干和 SAM 参数，新建跨视角融合/对象存在/旋转头，并重置 `detection_trained=False`。
迁移后使用 `oa-vla-train --resume /models/oa-vla-v2-init --stage detection ...` 训练配对，再按需联合训练。
自定义源架构需用 `--config exact_config.json` 指定完整尺寸。原点流 checkpoint 不能直接转换成此模型。
不传 tokenizer 时只导出模型和元数据，后续 policy 加载仍需提供 tokenizer 或从本地 Hugging Face 缓存获取。

## 命令行推理

```bash
oa-vla-predict --model /models/oa-vla --device cuda --prompt 'avoid obstacle' \
  --front front.png --side side.png --right-wrist right.png --left-wrist left.png \
  --state 0 0 0 0 0 0 0 0 0 0 0 0 --output prediction.json
```

所有入口也可用 `python -m oa_vla.train` / `.predict` / `.convert` / `.annotate`。该推理入口不连接机器人。

## TCP socket 服务与发送脚本

```bash
python -m pip install -e '.[serve]'
oa-vla-serve --model /models/oa-vla --device cuda --host 127.0.0.1 --port 8000
```

默认仅监听本机 `127.0.0.1:8000`。服务不打开相机或机器人。在另一个终端发送四张本地图像：

```bash
oa-vla-send --host 127.0.0.1 --port 8000 \
  --front front.png --side side.png --right-wrist right.png --left-wrist left.png \
  --state 0 0 0 0 0 0 0 0 0 0 0 0 --prompt 'avoid obstacle' \
  --output reply.json
```

图片路径须实际存在；状态须与训练时的维度及定义一致。回复包含 `server_metadata`、`context` 和
`prediction`，后者包含 `actions`、`objects` 和按视角展开的 `detections`。
发送入口也可使用 `python -m oa_vla.send`。

没有正式权重时，先创建仅用于测试的 tiny 随机模型，再启动 CPU 服务：

```bash
oa-vla-init-test --output /tmp/oa-vla-test
OMP_NUM_THREADS=1 oa-vla-serve --model /tmp/oa-vla-test --device cpu --num-steps 1
```

测试目录必须尚不存在；正式服务与测试服务使用同一端口时应分开启动。
TCP 协议使用长度前缀 MessagePack，携带请求/帧/时间戳/尺寸 context 并原样回传，
支持原始 RGB、PNG 或 JPEG，响应同时包含动作与配对 OBB。

训练导出会保留已声明的动作布局、来源和周期；不能仅凭动作维度推断绝对值或增量。
未训练 tiny 模型标记 `test_only=true`，返回动作张量和空 OBB，检测状态为 `not_trained`，硬件客户端拒绝执行。
本目录仍可独立复制运行 socket 服务/文件发送，不依赖父项目。
父项目的共享相机、Voxblox/椭球优化及机械臂接入见 [完整指南](../docs/OA_VLA_SOCKET_RUNTIME.md)。

## CPU 验证

```bash
OMP_NUM_THREADS=1 python examples/smoke_test.py
OMP_NUM_THREADS=1 python -m pytest -q
```

smoke test 使用完全随机的小模型和合成数据，执行真实单步训练、导出、重载和四视角推理，无需网络或权重。
默认生产架构并未因此得到训练或实机验证。包内测试还会将源码复制到临时目录，在禁止导入
OpenPI、父项目、JAX、Flax 和 pytest 的子进程中执行模型反传与推理。

## 来源与许可

oavla 由已有 PI0.5 风格 action expert 和 SAM 3 检测适配实现独立化而来。
OpenPI / Transformers 衍生代码保留 Apache-2.0 及原版权声明；SAM 3 模块受其单独 SAM License 约束，
Gemma 权重受 Gemma 条款约束。完整来源与许可见 `src/oa_vla/vendor/NOTICE.md` 及同目录许可证。
