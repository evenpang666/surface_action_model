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

实时检测的 `--workspace-bounds` 在左右臂各自的基座坐标系中应用，
经双臂标定转换后取并集；可用 `--right-workspace-bounds` 单独设置右臂。
默认每臂范围为 X ±0.85 m、Y -0.85～1.20 m、Z -0.12～0.70 m。
可视化仍可显示并集之外拍到的桌面点，但这些点不生成障碍物 OBB。

实时安全快照逐个跟踪聚类点；仅当两个 OBB 的交集超过较小框体积的 50% 时，
发布前优先合并为包络框，可用 `--obb-merge-overlap-ratio` 调整。默认需连续观测
3 帧才确认新簇；已确认的 OBB 仅在点簇变化达到 25% 且当前帧与前两帧
连续支持变化后更新，
距离判据为 15 mm。对应参数为 `--temporal-persistence-frames`、
`--obb-point-change-distance-m`、`--obb-point-change-fraction` 和
`--obb-change-confirm-frames`。完整快照只保留最新一帧，每条轨迹最多缓存最近
10 帧的点簇观察，可用 `--obb-history-frames` 调整。漏检时保留原框；原点簇
至少 60% 的点被原始深度明确测为后方空闲空间、连续 3 帧后提前移除旧框，
可用 `--obb-removal-confirm-frames` 调整连续帧数。连续漏检达到窗口长度时，
即使深度缺失或前景遮挡也删除旧框。
快照的 `diagnostics.obb_tracks` 显示每条轨迹的漏检帧数、缓存帧数和空闲证据比例，
`diagnostics.expired_obbs` 显示窗口到期删除的数量。
新簇、显著变化或漏检未厘清时，快照报告
`error: unstable obstacle point clusters`；安全快照仍包含已确认旧框和旧框未覆盖的
当前候选框，LeRobot 回放直接用这些框进行避障检查。
