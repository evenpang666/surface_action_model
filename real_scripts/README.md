# Real dual-UR7e runtime

当前主要入口：

- `live_obstacle_detection.py`：front + side D455、LingBot-Depth、双臂/双
  Robotiq 2F-85 剔除、桌面 OBB 和实测 current_surface 快照；全程只读。
- `live_obstacle_detection_viewer.py`：打开感知可视化。
- `run_ur7e_vla_safety_executor.py`：消费 Surface Action Model 输出和快照，
  执行 12-DOF CBF-QP 与 FK 轨迹复核；默认 dry-run。

双臂目前已断开，front 旧标定失效、side 未标定；配置阻止未标定在线运行。
新标定、TCP 核验和受控真机验收完成前不要启用执行。旧
`live_dual_ur7e_obstacle_model.py` 保留为实现/兼容入口。

共享固定点 FK 在 `dual_ur7e_surface.py`，CBF 几何在 `real_cbf_qp.py`。默认
配置与完整命令见仓库根目录 [README](../README.md)。

标定脚本、点云工具和可视化脚本仍作为当前硬件维护工具保留，但不再构成另一套
训练/部署主线。
