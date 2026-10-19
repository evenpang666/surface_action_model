# 双臂双视角标定

## 当前接入

项目使用 2026-10-17 的双臂双视角标定，文件已复制到项目内：

```text
configs/hardware/calibration/dual_arm_camera_transforms.yaml
```

入口为 `configs/ur7e_dual_quest3.yaml` 的 `safety.dual_arm_camera_calibration`。
四份源文件及标定观测记录均在同目录。汇总文件通过相对路径引用四份源文件，
并保留源文件的 SHA256 校验。

| 源标定 | camera serial | touch RMS |
| --- | --- | --- |
| left/front | 260922305390 | 2.206 mm |
| left/side | 260922303231 | 2.206 mm |
| right/front | 260922305390 | 1.955 mm |
| right/side | 260922303231 | 1.955 mm |

touch RMS 限值为 2.5 mm；相机质量、板运动、平移 spread 检查均通过。
重新计算的基座闭环误差为 7.077 mm / 0.1852°，相机闭环误差为
5.650 mm / 0.1852°，低于汇总的 10 mm / 3° 限值。
`camera_calibration_valid=true` 表示这组文件已经校验接入，不代替实机验收。

## 共用坐标图

`safety_module/dual_camera_calibration.py` 是统一读取入口；检测、双臂固定点
和独立随机路径验证都使用它。所有矩阵遵守 `parent_T_child`，把 child 点
转换到 parent。`left_base_to_right_base` 的 parent 是 left_base、child 是
right_base，因此用于把右臂点变换到左臂，**不需要取逆**。

```text
left_base_T_front = left/front 源标定矩阵
left_base_T_side  = left_base_T_front @ front_camera_to_side_camera.matrix
left_base_T_right = left_base_to_right_base.matrix

left_arm_point  = left_arm_FK(q_left) @ link_local_point
right_arm_point = left_base_T_right @ right_arm_FK(q_right) @ link_local_point
obstacle_point  = left_base_T_camera @ depth_unproject(pixel, depth)
```

使用汇总的相对矩阵并以 left/front 为锚点，保证运行时坐标图严格闭合。
合成 side 外参与独立 left/side 测量相差约 2.825 mm / 0.0926°，在质量限值内。
加载器重算四份源标定的闭环，不仅信任 `passed=true`；源文件 SHA256 不一致、
旋转不合法、相机序列号不匹配或质量超限时拒绝加载。

检测以实际请求流的内参、畸变参数和模型为准，避免不同分辨率下套用旧内参。
深度反投影、机器人深度渲染、实测表面点 lifting 使用一致的镜头畸变处理。
Inverse Brown 投影/反投影按
[RealSense SDK 实现](https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp)
处理，并有与已安装 SDK 的数值对比测试。

机器人剔除与固定点计算使用同一 `flange_to_active_tcp` 和 Robotiq 模型。
实测 active TCP 与 FK 相差超过 5 mm / 1° 时检测停止，防止探针 TCP 或
其他工具设置与部署几何不一致。`real_scripts/read_dual_active_tcp_calibration.py`
可以只读核验；实际 TCP 改变时应核验后更新共享机器人配置。

## 验证与启动

先做文件和几何校验，不连接机械臂或相机：

```bash
python scripts/validate_project_pipeline.py
python -m pytest -q
```

再启动只读检测，检查双视角点云、机器人剔除和障碍物坐标：

```bash
python real_scripts/live_obstacle_detection.py --no-browser
python real_scripts/live_obstacle_detection_viewer.py
```

快照 `calibration.sha256` 为汇总文件 SHA256。随机路径验证和 VLA 执行器
拒绝与固定点采样器标定哈希不同、或缺少哈希的实时快照。
修改标定配置后需重启检测服务，并重新生成路径。

相对基座变换改变会改变双臂 surface_model_hash；旧路径、旧双臂 shard 和
checkpoint 需要按新几何重建，不能放宽 hash 检查。相机移动后需重新标定。

## 保留的旧安装流程

`scripts/install_surface_calibration.py` 仍支持两份旧 `camera_calibration.json`
会话：touch RMS <=3 mm、重投影 RMS <=0.8 px，并拒绝覆盖已有外参文件。
该流程需要会话声明的固定板和相机条件，不是当前四份 YAML 的安装入口。
它会移除之前的汇总文件引用，避免混用不同会话。

本次仅完成文件、几何、模拟感知和离线回归验证，未发送运动或夹爪命令。
