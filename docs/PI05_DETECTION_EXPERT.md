# PI0.5 三头模型：action、point、detection

## 可行性与边界

现有 PI0.5 的 `embed_image` 返回按图像 patch 排序的空间 token，可作为二维障碍物框检测的预训练视觉底座。三头版本 `SurfaceActionDetectionModelPytorch` 继承现有 action + surface point 模型，仅增加检测 expert；PI0.5 的视觉权重仍冻结。front/side 分别经过**同一套**视觉权重，检测输入不占用 `base_0_rgb` / wrist 的 action-policy 相机槽。检测目标是当前帧 RGB 像素上的 2D OBB，不是三维碰撞框，也不预测未来障碍物轨迹。

这是一版可验证的结构，不预设视觉性能：PI0.5 图像塔采用 224×224 输入，可能丢失细小障碍物。必须按障碍物尺寸分层测召回率；若小目标达不到要求，再换高分辨率视觉底座或多尺度裁片，保持检测输出和标注契约不变。两路图像编码增加显存和推理时延，部署前须实测。SAM 3 可以作为未来的视觉底座或离线教师，但本版无需安装 SAM 3，也不改变现有动作 checkpoint 的视觉输入分布。

## 模型

```text
front RGB / side RGB → 冻结的同一个 PI0.5 image encoder → 空间 token
    → Linear(PI0.5 width, 256) + 位置嵌入
    → 3 层 Transformer query decoder；每视角 32 queries
    → objectness 与 (cx, cy, w, h, cos 2θ, sin 2θ)

PI0.5 VLM + action expert → action chunk
PI0.5 context + 当前表面点 + action chunk → point DiT → surface offsets
```

`cx,cy,w,h` 在保持纵横比缩放并填充到 224×224 后归一化。标注先规范为 `w≥h`，将交换宽高后的角度加 90°，避免同一个框出现两种监督表达。`θ` 是 RGB 图像坐标中顺时针为正的 OBB 角；二倍角编码使 θ 和 θ+180° 等价。输出服务再逆变换为原始 RGB 像素坐标。front/side query 各自匹配标注；`object_id` 只用于标注一致性校验，不当作物体类别，也不强制跨视角预测相同 query 编号。

## 联合训练目标

给定当前观测 `o`、动作 chunk `a`、当前机器人表面点 `p`、未来点位移 `d`，两视角人工框集合 `B_front,B_side`：

```text
L_total = L_action + λ_point L_point + λ_detection L_detection
L_action = masked_mean ||v_action(x_t,t,o) - (ε_action-a)||²
L_point  = masked_mean ||v_point(y_t,t,p,o,a) - (ε_point-d)||²
L_detection = L_objectness + L_matched_box
L_matched_box = SmoothL1(cx,cy,w,h) + 0.5·aspect·(1-cos(2(θ_pred-θ_gt)))
```

检测 query 与每视角 GT 用最小代价一对一匹配；未匹配 query 监督为背景，背景项权重 0.1。`aspect=|w-h|/(w+h)`（下限 0.05）减轻接近正方形框的角度歧义。`annotation.reviewed=0` 的视角 **不计算检测 loss**；人工确认的 `no_obstacles` 是有效背景样本。动作和点的 mask 沿用现有实现。默认 `λ_point=10`、`λ_detection=1`；权重必须以验证集的动作、点误差及检测召回共同调参。当前 OBB 目标没有可微旋转 IoU 项，第一版以框参数与周期角损失训练，评估时必须另算旋转 IoU。

## 两种训练算法

**联合训练**：加载 PI0.5 基础权重，冻结 VLM，更新 action expert、point DiT 和 detection expert。每个 batch 计算三个 loss；没有人工审核的样本仍可贡献 action/point loss。命令示意：

```bash
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/three_head_joint.pt \
  --detection-stage joint --point-loss-weight 10 --detection-loss-weight 1
```

**分步训练**：第 1 阶段训练 action + point，冻结 VLM 和 detection expert；第 2 阶段从第 1 阶段 checkpoint 加载权重，冻结 VLM、action expert、point DiT，仅在人工审核帧训练 detection expert。第 2 阶段保留 checkpoint 的动作/状态归一化参数和相机映射，不重新定义动作语义。也可以直接从现有 schema 3 的已训练 action + point checkpoint 进入第 2 阶段。

```bash
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/three_head_stage1.pt \
  --detection-stage action_point

python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/three_head_stage2.pt \
  --detection-stage detection --resume outputs/models/three_head_stage1.pt
```

训练器通过 shard 中的源 HDF5 路径、`sample_frame_indices` 和 `timestamps_ns` 对齐 v2 RGB OBB 标注。`--validate-only` 可在加载 PI0.5 前检查样本及已审核帧数。schema 4 checkpoint 可由策略服务加载；只有 `joint` 或 `detection` 阶段完成的 checkpoint 才标记为检测可用。服务收到 front 与 side RGB 时返回两路检测框；缺少 side 时显式返回 `detection_status=missing_side_rgb`。现有执行器尚未发送 side RGB，因此三头检测输出暂不进入在线安全决策。二维框若要参与避障，仍须结合对齐深度与标定生成三维证据并验证时效。
