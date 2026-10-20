# 双臂双视角标定

检查与部署在仓库根目录、`surface-runtime` 环境执行。先停止占用 front/side 的进程，
并在 PolyScope 完成探针/工具 TCP 标定；脚本只读取活动 TCP，不会自动标定工具或移动机器人。

## 检查现有标定

```bash
python scripts/validate_project_pipeline.py
python real_scripts/read_dual_active_tcp_calibration.py
```

部署配置：`configs/ur7e_dual_quest3.yaml`。
汇总文件：`configs/hardware/calibration/dual_arm_camera_transforms.yaml`；四份源标定在同目录。
TCP 测量默认写入 `outputs/calibration/dual_flange_to_active_tcp.json`，
核验后手动更新项目配置的 `robot.flange_to_active_tcp`，不会自动安装。

## 重新采集

采集入口位于相邻的 `../quest3_collect` 仓库，需切换到该项目原标定环境，
要求 `cv2.aruco.CharucoBoard_create` 可用；新版 OpenCV 的运行环境不能直接运行此旧采集入口。
实体板参数：11×8 格、格边长 70 mm、
marker 52 mm、`DICT_5X5_100`；必须与实际印刷板一致。

先新建会话，避免已通过的旧结果被自动跳过：

```bash
mkdir -p outputs/calibration/dual_session
cp configs/hardware/hardware_dual_teleop.yaml outputs/calibration/dual_session/hardware.yaml
PYTHONPATH=../quest3_collect python -m quest3_robot_data.calibrate_dual_camera_charuco \
  --config outputs/calibration/dual_session/hardware.yaml \
  --squares-x 11 --squares-y 8 --square-size-m 0.070 --marker-size-m 0.052 \
  --dictionary DICT_5X5_100 --max-touch-rms-mm 2.5
```

每次新标定换一个会话目录。开始前核对 hardware YAML 中的机器人 IP 和相机序列号。

1. 固定两台相机；把板放到左臂可达且两路可见的位置。
2. 按窗口/终端提示锁定板观测，用已校准 TCP 依次触碰指定内角，回车记录。
3. 移开机械臂，复核板未移动。
4. 按提示对右臂重复；左右臂可使用不同板位置和触点，但相机保持固定。

板仍固定时，可在原命令后加 `--arm left --resume` 或 `--arm right --resume` 续做。
板移动后不要复用旧触点。输出位于会话目录的 `calibration/`，包含四份相机 YAML、观测、触点和汇总 YAML。

## 接入新结果

切回 `surface-runtime`，把项目 YAML 的下列路径指向新会话：

```yaml
quest3:
  hardware_config: outputs/calibration/dual_session/hardware.yaml
safety:
  dual_arm_camera_calibration: outputs/calibration/dual_session/calibration/dual_arm_camera_transforms.yaml
  front_calibration: outputs/calibration/dual_session/calibration/left_base_to_front_camera.yaml
  side_calibration: outputs/calibration/dual_session/calibration/left_base_to_side_camera.yaml
  right_front_calibration: outputs/calibration/dual_session/calibration/right_base_to_front_camera.yaml
  right_side_calibration: outputs/calibration/dual_session/calibration/right_base_to_side_camera.yaml
  camera_calibration_valid: true
```

这是现有配置的字段更新示例，保留其余字段。保留汇总引用的源文件，勿手改矩阵或 SHA256。

```bash
python scripts/validate_project_pipeline.py
python real_scripts/live_obstacle_detection.py --robot-mask
```

检查双视角点云重合、机械臂模型与深度位置一致。TCP 检查阈值为 5 mm / 1°；
出现 `Active TCP differs` 时核对工具配置，修正后重启检测。
基座变换改变后重新生成相关 shard、路径和 checkpoint；相机移动后重新标定并重做障碍标注。

单臂标定见 [UR7e + Pika](SINGLE_UR7E_PIKA.md)。旧 JSON 会话的安装工具为
`scripts/install_surface_calibration.py`，不能用它安装当前四份 YAML。
