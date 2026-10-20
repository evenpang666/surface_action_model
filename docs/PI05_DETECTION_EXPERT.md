# PI0.5 action / point / detection 三头训练

在 `surface-policy` 环境运行。先准备[实测点 shard](IMPLEMENTATION_GUIDE.md#数据预处理)与
[HDF5 RGB OBB 标注](BATCH_OBSTACLE_ANNOTATION.md)，保持 shard 引用的源 HDF5 可读。

## 联合训练

```bash
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/three_head_joint.pt \
  --detection-stage joint
```

先检查数据时加 `--validate-only`。基础权重默认读取项目配置的 `model.pretrained`。

## 分阶段训练

先训练动作与表面点，再只训练检测分支：

```bash
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/three_head_stage1.pt \
  --detection-stage action_point
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/three_head_stage2.pt \
  --detection-stage detection --resume outputs/models/three_head_stage1.pt
```

第 2 阶段也可加载已训练的 schema 3 action + point checkpoint。
未人工审核的帧不计算 detection loss；已确认无障碍的帧用于背景监督。

| 参数 | 默认 / 用法 |
| --- | --- |
| `--point-loss-weight` | 10 |
| `--detection-loss-weight` | 1 |
| `--detection-stage none` | 普通 action + point 模型 |
| `--resume PATH` | 加载前一阶段 checkpoint |

## 服务与输出

```bash
python scripts/serve_quest3_pi05_safety.py --checkpoint outputs/models/three_head_stage2.pt
```

服务请求需包含 front 与 side RGB，输出为各自原始 RGB 像素坐标中的二维旋转框。
缺少 side 会返回 `detection_status=missing_side_rgb`。
当前实机执行器尚未发送 side RGB，检测结果尚未接入在线避障；实时障碍仍来自 RGB-D 检测。
