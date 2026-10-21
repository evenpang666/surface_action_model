# 环境、数据与模型运行

当前配对 OBB 模型与训练/标注入口见 [独立 OA-VLA](../oa-vla/README.md)，配置 format 2。
本文下述 schema-5 集成版是历史独立水平框实现，不包含跨视角对象配对；旧权重须迁移并再训练。

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

当前策略只需要四视角 RGB、机器人状态、动作和 front/side 的人工审核框标注，无需深度、CoTracker 或表面点。
在 `surface-runtime` 环境运行：

```bash
python scripts/preprocess_quest3_rgb_action.py \
  --episode /path/to/episode.hdf5 --output outputs/rgb_action/episode_00000.shard
```

原 episode 必须包含 `observations/images/{front,side,right_wrist,left_wrist}/rgb`，
并与状态、时间戳共享帧轴。缺少视角会报错，不能用零图替代。四个相机可使用不同分辨率。
现有四视角 `.npz` / `.shard` 也可直接训练；其中旧点云字段被忽略。
标注沿用[RGB OBB 标注流程](BATCH_OBSTACLE_ANNOTATION.md)，训练时转换为外接轴对齐 BB。
预处理会保留源 HDF5 副本，便于定位标注；旧深度与点流流程见 `scripts/preprocess_quest3_hdf5.py`。

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
python scripts/train_pi05_sam3_pytorch.py \
  --dataset outputs/rgb_action --output outputs/models/sam3_vla.pt --validate-only
python scripts/train_pi05_sam3_pytorch.py \
  --dataset outputs/rgb_action --output outputs/models/sam3_vla.pt \
  --sam3-checkpoint /path/to/sam3.pt
python scripts/serve_quest3_pi05_safety.py --checkpoint outputs/models/sam3_vla.pt
```

默认联合训练动作 expert、SAM 3 检测解码器及适配层，冻结 PaliGemma。动作分支使用四视角；
检测分支通过独立 front/side/text prefix 保证腕部图像不会间接泄漏。详见[模型文档](PI05_DETECTION_EXPERT.md)。
`--tokenizer-model PATH` 可指定离线 tokenizer；`--precision float16` 适用于 V100。
`--gradient-accumulation-steps`、`--max-steps` 和 `torchrun` 多卡入口继续支持。

服务请求必须包含四路 RGB，格式见 [README](../README.md#3-策略服务)。
新模型不返回表面点，因此不能接入旧 `run_ur7e_vla_safety_executor.py` 的点流 CBF 协议。
旧 checkpoint 使用 `scripts/serve_quest3_pi05_surface_legacy.py`；标定、实时 RGB-D 地图与独立回放功能继续使用原入口。

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
