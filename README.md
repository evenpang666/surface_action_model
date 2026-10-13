# Surface Action Model

基于 OpenPI PI0.5 的双臂动作与表面点联合预测项目。模型由 RGB、任务文本和机器人状态生成动作及未来表面点；在线运行时结合双视角深度、障碍物地图、CBF-QP 和 FK 轨迹复核。方法类为 `SurfaceActionModelPytorch`。

## 安装

使用 Python 3.11，在仓库根目录创建独立的策略和感知环境：

```bash
conda create -n surface-policy python=3.11 -y
conda run -n surface-policy bash -c 'cd requirements && python -m pip install -r policy.txt'
conda create -n surface-runtime python=3.11 -y
conda run -n surface-runtime bash -c 'cd requirements && python -m pip install -r runtime.txt'
conda run -n surface-runtime bash -c 'cd requirements && python -m pip install -r preprocess.txt'
conda run -n surface-runtime python -m pip install --no-deps -e ./lingbot-depth
```

原始 HDF5 预处理还需准备 CoTracker 源码与权重。在线感知需要 RealSense 驱动与 USB 权限；连接机械臂需要 UR 网络。依赖、标定和硬件说明见[实现与运行细节](docs/IMPLEMENTATION_GUIDE.md)。

## 快速使用

以下命令从仓库根目录执行。先按[数据预处理流程](docs/IMPLEMENTATION_GUIDE.md#数据预处理)将录制数据生成到 `outputs/surface_action/`，并准备 PI0.5 基础权重。HDF5 深度必须符合[已对齐深度契约](docs/QUEST3_DEPTH_ALIGNMENT.md)。
如需包含四路 RGB、CoTracker 点及后续 front/side RGB OBB 标注的标准 LeRobot 数据集，
使用[单独导出流程](docs/QUEST3_DEPTH_ALIGNMENT.md#四视角-lerobot-训练数据导出)；下方 PI0.5 训练命令仍直接读取 shard。

1. 验证数据并训练模型：

   ```bash
   conda run -n surface-policy python scripts/train_pi05_ur7e_surface_pytorch.py \
     --dataset outputs/surface_action --output outputs/models/surface_action.pt --validate-only

   conda run -n surface-policy python scripts/train_pi05_ur7e_surface_pytorch.py \
     --dataset outputs/surface_action --output outputs/models/surface_action.pt \
     --pretrained outputs/pretrained/pi05_base_pytorch/model.safetensors \
     --max-points 128 --batch-size 1 --precision bfloat16
   ```

2. 启动策略服务：

   ```bash
   conda run -n surface-policy python scripts/serve_quest3_pi05_safety.py \
     --checkpoint outputs/models/surface_action.pt --device cuda --port 8000
   ```

3. 完成[双臂双视角标定](docs/CALIBRATION.md)并核对硬件后，在测试端分别启动感知、可视化和一次默认 dry-run 执行：

   ```bash
   conda run -n surface-runtime python real_scripts/live_obstacle_detection.py --no-browser
   conda run -n surface-runtime python real_scripts/live_obstacle_detection_viewer.py
   conda run -n surface-runtime python real_scripts/run_ur7e_vla_safety_executor.py \
     --prompt "manipulate the tabletop instruments safely" --once
   ```

   三条命令分别占用终端。物理执行需额外的 `--execute` 和精确确认串；当前标定接入已完成离线校验，受控实机验收尚未进行。操作细节见[实现与运行细节](docs/IMPLEMENTATION_GUIDE.md#在线运行)。

## 常见问题

- **安装或依赖冲突**：策略与感知使用独立环境；ChArUco 需用同版本 `opencv-contrib-python` 替代普通 OpenCV wheel。见[环境说明](docs/IMPLEMENTATION_GUIDE.md#环境)。
- **HDF5 深度被拒绝**：确认采集端保存的是 RGB 对齐深度，尺寸及 topic/属性与录制 RGB 一致；原生深度不能交给该预处理入口。见[Quest3 已对齐深度契约](docs/QUEST3_DEPTH_ALIGNMENT.md)。
- **感知启动失败或 TCP 校验报错**：核对相机实际流配置、USB 连接、双臂 active TCP 与固定表面 FK 配置。见[在线运行与诊断](docs/IMPLEMENTATION_GUIDE.md#在线运行)。
- **快照过期或检测帧率低**：查看 `snapshot.json?compact=1` 的 `diagnostics` 和采集到发布耗时，再按[实时感知说明](real_scripts/README.md)排查。
- **默认没有物理动作**：执行器默认 dry-run；验收与执行条件见[在线运行](docs/IMPLEMENTATION_GUIDE.md#在线运行)。

## 详细文档

- [实现与运行细节](docs/IMPLEMENTATION_GUIDE.md)：硬件、环境、预处理、训练、在线运行与诊断。
- [标定](docs/CALIBRATION.md) · [原生深度对齐](docs/QUEST3_DEPTH_ALIGNMENT.md) · [批量障碍物标注](docs/BATCH_OBSTACLE_ANNOTATION.md)
- [实时感知](real_scripts/README.md) · [无模型障碍物验证](docs/MODEL_FREE_OBSTACLE_VALIDATION.md)
- [代码审查状态](VLAsafe_CODE_REVIEW.md) · [历史入口](docs/legacy/INDEX.md)
