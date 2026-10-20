# Surface Action Model

双 UR7e + Robotiq 2F-85：PI0.5 预测动作与机器人表面点，RGB-D 感知提供障碍物地图。
所有命令在仓库根目录执行，示例中的数据路径按实际文件替换。

## 1. 环境配置

Python 3.11，策略与感知分别安装：

```bash
conda create -n surface-policy python=3.11 -y
conda run -n surface-policy bash -c 'cd requirements && python -m pip install -r policy.txt'
conda create -n surface-runtime python=3.11 -y
conda run -n surface-runtime bash -c 'cd requirements && python -m pip install -r preprocess.txt'
```

回放还需 LeRobot、OSQP 和原生 Voxblox；安装命令见[环境与数据准备](docs/IMPLEMENTATION_GUIDE.md#环境)。
后文训练/策略服务使用 `surface-policy`；感知、预处理、回放使用 `surface-runtime`。
标定环境与采集入口见[标定说明](docs/CALIBRATION.md)。

## 2. 训练

先按[数据预处理](docs/IMPLEMENTATION_GUIDE.md#数据预处理)生成 `outputs/surface_action/`，
准备 `outputs/pretrained/pi05_base_pytorch/model.safetensors`。

```bash
conda activate surface-policy
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/surface_action.pt
```

默认读取项目配置中的 PI0.5 权重，batch size 为 1、bfloat16、每帧最多 128 个点。
只检查数据加 `--validate-only`；[三头模型训练](docs/PI05_DETECTION_EXPERT.md)。

## 3. 推理

策略机启动服务：

```bash
conda activate surface-policy
python scripts/serve_quest3_pi05_safety.py --checkpoint outputs/models/surface_action.pt
```

控制机分别在两个终端运行：

```bash
conda activate surface-runtime
# 终端 A：为默认 visual-point 模型提供实测表面点
python real_scripts/live_obstacle_detection.py --robot-mask --no-browser
# 终端 B：推理一次，默认不发送运动
python real_scripts/run_ur7e_vla_safety_executor.py --prompt "抓取器械并放入托盘" --once
```

服务默认端口 `8000`，策略机地址在 `configs/ur7e_dual_quest3.yaml` 的 `safety.policy_host` 配置；
也可给执行器传 `--policy-host IP`。连续运行去掉 `--once`。
实机执行参数及夹爪开关见[在线运行](docs/IMPLEMENTATION_GUIDE.md#在线运行)。

## 4. 标定

在 `surface-runtime` 环境中核验现有标定及当前活动 TCP：

```bash
python scripts/validate_project_pipeline.py
python real_scripts/read_dual_active_tcp_calibration.py
```

项目配置：`configs/ur7e_dual_quest3.yaml`；标定目录：`configs/hardware/calibration/`。
[重新采集双臂双视角标定](docs/CALIBRATION.md) · [单臂 Pika 标定](docs/SINGLE_UR7E_PIKA.md)。

## 5. 障碍可视化

```bash
conda activate surface-runtime
python real_scripts/live_obstacle_detection.py
```

自动打开 <http://127.0.0.1:8766>。只看重建、不连接机器人加 `--no-robot`；
Voxblox 地图加 `--occupancy-backend voxblox`。

| 按键 | 操作 |
| --- | --- |
| B | 体素筛选，默认关闭 |
| O / L | 障碍椭球 / 连杆椭球，默认关闭 |
| V / R | 体素框 / 机械臂显示 |
| 空格 | 暂停 / 继续显示 |

B 关闭时，障碍椭球使用全部当前占据体素；B 开启时，仅使用筛选体素。
双臂启动时加载 `outputs/manual_obstacles/` 下目录修改时间最新的 `annotations.json`，供 B 开启时使用。
默认关闭机器人深度遮罩，需要剔除机器人时加 `--robot-mask`。
[地图后端、参数与排错](real_scripts/README.md)。

## 6. 回放避障

无需模型权重。在 `surface-runtime` 环境中先转换数据并保存障碍标注：

```bash
python -m obstacle_validation.lerobot_replay convert \
  --hdf5 /path/to/episode.hdf5 --output outputs/obstacle_validation/my_episode_lerobot
python real_scripts/manual_obstacle_annotation.py
```

标注页点击捕获，在 front/side 为同一物体画同编号框，保存。退出其他占用相机的进程后回放：

```bash
python -m obstacle_validation.lerobot_replay replay \
  --dataset outputs/obstacle_validation/my_episode_lerobot --mode avoid
```

**此命令默认实机执行。** 自动使用最新 `annotations.json`，输出至
`outputs/obstacle_validation/annotation_replay_时间戳/`。
虚拟回放加 `--no-execute`；不筛选、不避障改用 `--mode baseline`。
复位阶段不避障；避障阶段允许距离约束松弛。
[回放参数与双终端方式](docs/MODEL_FREE_OBSTACLE_VALIDATION.md)。

## 文档

- [环境、数据预处理、权重、训练与部署](docs/IMPLEMENTATION_GUIDE.md)
- [标定](docs/CALIBRATION.md) · [单臂 UR7e + Pika](docs/SINGLE_UR7E_PIKA.md)
- [RGB 障碍标注](docs/BATCH_OBSTACLE_ANNOTATION.md) · [深度与 LeRobot 导出](docs/QUEST3_DEPTH_ALIGNMENT.md)
- [实时感知](real_scripts/README.md) · [回放避障](docs/MODEL_FREE_OBSTACLE_VALIDATION.md)
- [实验记录与评估](docs/EXPERIMENT_TABLE_DESIGN.md) · [历史入口](docs/legacy/INDEX.md)
