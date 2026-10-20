# 环境、数据与模型运行

命令均在仓库根目录执行；快速入口见 [README](../README.md)。

## 环境

```bash
conda create -n surface-policy python=3.11 -y
conda run -n surface-policy bash -c 'cd requirements && python -m pip install -r policy.txt'
conda create -n surface-runtime python=3.11 -y
conda run -n surface-runtime bash -c 'cd requirements && python -m pip install -r preprocess.txt'
```

`surface-policy` 用于训练和策略服务；`surface-runtime` 用于预处理、标定、感知和回放。
已有 `safety` 环境可继续使用。系统需配置 RealSense USB 权限和 UR 网络。

标定需要 ChArUco；在标定环境中替换普通 OpenCV，避免混装：

```bash
conda activate surface-runtime
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python==4.10.0.84
```

LeRobot 回放/导出使用与仓库 OpenPI 相同的 LeRobot 版本：

```bash
python -m pip install osqp 'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
sudo apt-get install build-essential cmake git libeigen3-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler
python scripts/build_voxblox.py
```

切换 Python 环境后需为该环境重新构建 Voxblox。Pika 夹爪读取另需 `python -m pip install pyserial`。
可选 LingBot-Depth：`python -m pip install --no-deps -e ./lingbot-depth`；默认不加载。
安装完成可运行 `python -m pip check` 检查依赖。

## 数据预处理

使用 episode 采集时的标定；front 深度必须已对齐到 RGB，要求见[深度说明](QUEST3_DEPTH_ALIGNMENT.md)。
在 `surface-runtime` 环境依次运行，每个 episode 重复一次：

```bash
python scripts/preprocess_quest3_hdf5.py \
  --episode /path/to/episode.hdf5 \
  --output outputs/gates/episode_00000.shard --geometry-only

python scripts/preprocess_cotracker_surface.py \
  --episode /path/to/episode.hdf5 \
  --geometry-shard outputs/gates/episode_00000.shard \
  --calibration configs/hardware/calibration/left_base_to_front_camera.yaml \
  --output outputs/tracks/episode_00000.npz \
  --preview outputs/tracks/episode_00000.png \
  --visualization outputs/tracks/episode_00000.html

python scripts/preprocess_quest3_hdf5.py \
  --episode /path/to/episode.hdf5 \
  --measured-flow-npz outputs/tracks/episode_00000.npz \
  --output outputs/surface_action/episode_00000.shard
```

CoTracker 入口默认从以下位置读取源码与权重；位置不同时传 `--repo` / `--checkpoint`：

```text
~/.cache/torch/hub/facebookresearch_co-tracker_main/
~/.cache/torch/hub/checkpoints/scaled_offline.pth
```

检查 PNG/HTML 中跟踪点是否贴合机器人。训练读取 `.shard`，不是回放用的绝对关节 LeRobot 数据集。
不规则时间间隔需要先重采样；不要混用不同标定或动作周期的数据。

## PI0.5 基础权重

准备 OpenPI `pi05_base` JAX checkpoint 到 `~/.cache/openpi/openpi-assets/checkpoints/pi05_base/`，
在 `surface-policy` 环境转换：

```bash
python openpi/examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --config-name pi05_aloha --output-path outputs/pretrained/pi05_base_pytorch \
  --precision bfloat16
```

训练默认从项目 YAML 的 `model.pretrained` 读取生成的 `model.safetensors`。
其他位置用 `--pretrained /path/to/model.safetensors`。Transformers 扩展由仓库运行时加载，无需覆盖 site-packages。

## 训练与策略服务

```bash
conda activate surface-policy
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/surface_action.pt --validate-only
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/surface_action.pt
python scripts/serve_quest3_pi05_safety.py --checkpoint outputs/models/surface_action.pt
```

| 参数 | 用法 |
| --- | --- |
| `--epochs 20` | 训练轮数 |
| `--batch-size 1` | 每进程 batch size |
| `--precision bfloat16` | 默认精度；不支持时按硬件改为其他可用精度 |
| `--max-points 128` | 每帧训练点数上限 |
| `--gradient-accumulation-steps N` | 梯度累积 |
| `--max-steps N` | 限制优化器更新次数 |
| `--tokenizer-model PATH` | 离线 tokenizer 路径；训练和服务均支持 |

默认训练 action + point，冻结 VLM。检测分支训练见[三头模型](PI05_DETECTION_EXPERT.md)。
默认策略只使用 front RGB；四视角 LeRobot 导出不会自动增加模型输入视角。

## 在线运行

先完成[标定检查](CALIBRATION.md)。控制机在 `surface-runtime` 的两个终端运行：

```bash
# 终端 A：visual-point 模型需要 --robot-mask 才有实测 current_surface
python real_scripts/live_obstacle_detection.py --robot-mask --no-browser
# 终端 B：一次 dry-run
python real_scripts/run_ur7e_vla_safety_executor.py --prompt "抓取器械并放入托盘" --once
```

策略服务默认 `8000`；执行器从 `safety.policy_host` 读取地址，可用 `--policy-host IP` 覆盖。
去掉 `--once` 连续运行。实机运行：

```bash
python real_scripts/run_ur7e_vla_safety_executor.py --prompt "抓取器械并放入托盘" \
  --execute --execute-ack I_UNDERSTAND_DUAL_UR7E_WILL_MOVE
```

同时控制双夹爪加 `--execute-grippers`。只读 TCP 工具不会自动更新 YAML 中的 TCP 矩阵。

## 排错

| 问题 | 处理 |
| --- | --- |
| 缺少实测表面点 | 检测端加 `--robot-mask`，检查快照 `current_surface` |
| 相机被占用 | 退出其他检测/标注进程，再启动；标注捕获完成后会释放相机 |
| `Active TCP differs` | 核对控制器活动 TCP 与 `robot.flange_to_active_tcp`，修正后重启检测 |
| 标定哈希不一致 | 核对共享标定；变换改变后重建相关 shard、路径与 checkpoint |
| 快照过期 | 查看 `diagnostics.capture_to_publish_seconds` 和 `frame_interval_seconds`，排查 USB/分辨率/处理耗时 |
| HDF5 深度被拒绝 | 检查 RGB 对齐声明、尺寸和深度单位，见[深度说明](QUEST3_DEPTH_ALIGNMENT.md) |

快照：<http://127.0.0.1:8766/snapshot.json?compact=1>。
离线检查用 `python scripts/validate_project_pipeline.py`；回归用 `python -m pytest -q`。
