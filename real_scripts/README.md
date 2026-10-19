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
配置与完整命令见[实现与运行细节](../docs/IMPLEMENTATION_GUIDE.md)。

标定脚本、点云工具和可视化脚本仍作为当前硬件维护工具保留，但不再构成另一套
训练/部署主线。

实时检测的 `--workspace-bounds` 在左右臂各自的基座坐标系中应用，
经双臂标定转换后取并集；可用 `--right-workspace-bounds` 单独设置右臂。
默认每臂范围为 X ±0.85 m、Y -0.85～1.20 m、Z -0.12～0.70 m。
可视化仍可显示并集之外拍到的桌面点，但这些点不生成障碍物 OBB。

默认安全后端为局部概率占据地图（`obstacle_backend=probabilistic_voxels`），
用 `--occupancy-map` 显式开启，`--no-occupancy-map` 关闭：

```bash
python real_scripts/live_obstacle_detection.py --no-browser --no-lingbot-depth --no-occupancy-map
```

关闭后使用 `obstacle_backend=clustered_obbs`：过滤点云经 DBSCAN、桌面约束 OBB、
时序匹配生成安全框，当前未确认/变化簇也有候选框覆盖。不创建或更新体素地图，
不生成完整像素地图终点，不发布 `occupancy_map` 或地图诊断，消费者只检查 OBB
与双臂几何，不执行体素 unknown/过期空闲的扫掠检查。深度噪声过滤和有界桌面
先验继续生效；已观测障碍漏检立即撤下旧框、仅转 unknown，仍整体阻止运动，
连续可信原始空闲观测才能解除。可疑原始返回不能作为清除证据，也会否决其他视角
对旧障碍见证点的空闲判断。新可疑点不形成体素 unknown 区域，因为此模式没有地图。
此模式 `--point-stride`、`--cluster-eps-m`、`--cluster-min-samples`、`--min-cluster-points`、
`--attachment-distance-m` 及 OBB 时序/合并参数生效；默认簇点数门槛为 80，
稀疏采样或高门槛可能漏掉细小目标。`--occupancy-*` 地图参数仅在地图开启时生效。

地图开启时使用 NumPy 有界体素网格，不需要 ROS 或额外映射依赖。地图固定在首次桌面先验的
坐标轴上，覆盖双臂工作区并集的桌面上方范围；地图外为 `outside`，属于此局部
检测器覆盖范围之外，不代表被证实空闲。桌面以下和未配置区域不提供环境安全保证。
检测包含悬空物体，默认无 0.35 m 上限；`--min-height-m` 和工作区边界仍限制范围。
安全几何不再依赖 DBSCAN 的簇编号、密度或最小点数。

每个体素保存 log-odds、最近原始/补全证据来源和观测时间。一个完整双视角帧中，
同一体素最多更新一次，同帧占据证据优先于空闲证据。已接受原始非机器人深度终点
立即成为占据体素，即使该位置此前有较强空闲信念；有效深度之前的射线路径提供
空闲证据，终点、终点后方、无效深度和机器人遮挡后方不会被射线清空。
经双视角核验的补全只能增加占据证据，不能产生空闲射线。

两种后端的原始深度均默认先过滤噪声：3×3 像素邻域内至少 2 个有效非机器人邻点，深度差
不超过 8 mm + 0.005×较近深度，三维间距不超过 20 mm。保留原始值，不跨深度
边缘平滑；一像素宽细线的内部也可得到支持。仅有 1 个邻点时需连续 3 帧一致
观测确认；缺失、深度变化、相机标定/分辨率变化或超过 2 秒间隔重置确认。
完全孤立点不能仅凭持续出现确认，但可由另一视角独立接受的原始表面核验；
两个未确认噪点不会相互确认。渲染机器人后方的矛盾返回也被拒绝，前方物体保留。
拒绝深度既不生成 OBB，也不产生空闲射线。地图开启时，其终点以全像素分辨率写成 unknown，
即使不在 `--point-stride` 采样格上，也会撤销历史 free 并否决同帧空闲射线及补全。
已有可信原始占据可以覆盖可疑命中；单纯可疑点不登记历史障碍，但局部扫掠拒绝
进入其 unknown。已确认障碍的历史危险记录不因过滤而清除。
地图开启时，已接受原始终点同样使用完整像素证据生成占据；`--point-stride` 只影响原始显示/桌面
拟合点云与补全点采样，不会丢掉采样格之间已确认的小表面。

| 原始深度过滤参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--raw-depth-neighbor-pixels` | 1 | 支持邻域半径，默认 3×3 |
| `--raw-depth-min-neighbors` | 2 | 立即接受所需邻点数，不包括自身 |
| `--raw-depth-absolute-tolerance-m` | 0.008 m | 邻点及跨帧/跨视角深度差绝对容差 |
| `--raw-depth-relative-tolerance` | 0.005 | 随深度增加的容差系数 |
| `--outlier-neighbour-m` | 0.020 m | 支持邻点的三维距离上限，新地图中也生效 |
| `--raw-depth-confirm-frames` | 3 | 弱支持点的连续观测确认次数 |
| `--raw-depth-max-gap-s` | 2.0 s | 相邻确认观测允许的最大间隔 |

默认阈值需在真实相机距离和目标尺寸下验证；连续、系统性的深度偏差或标定残留
仍可能通过局部支持检查，不能仅靠加大过滤阈值解决。
诊断 `depth_evidence` 按相机给出 `spatial_supported_pixels`、`temporal_confirmed_pixels`、
`multiview_confirmed_pixels`、`isolated_pixels`、`robot_occluded_pixels`、
`accepted_raw_pixels`、`uncertain_raw_pixels`；过滤耗时为 `raw_depth_filter_seconds`，
地图可疑终点计数为 `diagnostics.occupancy.uncertain_raw_voxels`。

观测丢失的已占据体素立即转为 unknown，不发布或保留其历史 OBB。
历史危险记录不会因帧数、窗口或概率衰减而消失。只有原始射线连续空闲证据达到
`--obb-removal-confirm-frames`（默认 3 帧）且概率降至空闲阈值，才清除危险记录；
充分观测到终点则恢复 occupied。未观测体素始终为 unknown；缺少新观测的信念
以 1 秒半衰期趋向 0.5，而不会趋向空闲。已确认空闲体素也会在观测超时后转为 unknown。

`unknown_obstacles` 仅发布历史危险体素的数量和状态，不含历史中心、轴或尺寸；
这些 unknown 仍使回放和执行器整体停止。地图中从未观测或已过期的空闲体素，
在 RRT 边、FK 关节/夹爪组合扫掠和回放恢复检查中拒绝经过它们的动作。
运动起点的 unknown 体素同样不豁免，避免小步连续运动绕过未知空间检查；
因此机械臂自身剔除附近缺少环境证据时可能保守拒绝运动。纯保持不新增扫掠，
但仍须通过原有障碍物及双臂几何检查。`allow_tracked_obbs` 不会绕过地图 unknown。

占据体素在最多 4×4×4 体素的小块内合并为实心长方体 OBB，只合并全部占据的
长方体，不跨越空体素填充间隙。地图 OBB 使用 `--box-margin-m` 膨胀，并直接供
现有 CBF；膨胀后不再全局合并。CBF 修正后仍必须通过地图观测和 FK 扫掠复核。
网格最多 200 万体素、几何最多 8192 个框；超过预算会明确报错，不能静默丢弃障碍。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--occupancy-map` / `--no-occupancy-map` | 开启 | 开启局部概率地图，或关闭并使用聚类 OBB |
| `--occupancy-resolution-m` | 0.02 m | 固定体素边长；越小计算和内存成本越高 |
| `--occupancy-ray-stride` | 6 | 原始空闲射线的像素采样步长；已接受原始占据终点使用完整像素证据 |
| `--occupancy-free-margin-m` | 0.015 m | 在实测终点或更近的机器人表面前截断空闲射线 |
| `--occupancy-hit-probability` | 0.7 | 原始占据证据的更新概率 |
| `--occupancy-miss-probability` | 0.4 | 原始空闲证据的更新概率 |
| `--occupancy-free-probability` | 0.35 | 确认空闲的最大占据概率 |
| `--occupancy-free-ttl-s` | 2.0 s | 空闲证据最大年龄，包含采集到执行的时间 |
| `--occupancy-box-tile-voxels` | 4 | 每个框的最大块尺寸（膨胀前） |

地图开启时，快照 `occupancy_map` 在 compact/full/viewer 响应中均保留。`state_data` 为按 C 顺序
展平的 uint8（0 unknown、1 free、2 occupied、3 outside），`age_ms_data` 为小端 uint16，
`log_odds_data` 为小端 float32，`source_data` 为 uint8；都使用 zlib 后 base64 编码。
来源编码为 raw_hit=1、raw_free=2、verified_completion=4、uncertain_raw=8。
形状、原点、坐标轴和采集时间同时发布，消费者校验尺寸、状态计数、编码和时间。
空闲证据年龄在消费者端继续增加，重复读取同一快照不会延长有效期。
诊断见 `diagnostics.occupancy`、`occupancy_update_seconds`、`occupancy_boxes_seconds` 和
`occupancy_encode_seconds`。`diagnostics.occupancy_map_enabled` 在两种后端均发布。
地图开启时，旧聚类/OBB 时序参数保留命令行兼容性，但不控制地图几何；关闭时重新用于聚类与跟踪。
`--outlier-neighbour-m` 现在用于原始深度支持检查，必须为正值；
`--obb-removal-confirm-frames` 继续控制原始空闲证据的连续帧数。

桌面先验只使用原始实测点：启动时需暴露足够桌面，或用
`--table-plane NX NY NZ OFFSET` 提供 left_base 中 `n·x + offset = 0` 的已测平面。
在线仅拟合先验附近的点，支持不足时固定先验；相对启动锚点默认限制 3° 法向变化
及 10 mm 位移，以 0.1 权重缓慢更新。物体顶面不能替换桌面先验。
可通过 `--table-prior-min-inlier-fraction`、`--table-prior-max-angle-deg` 和
`--table-prior-max-offset-m` 调整；诊断见 `diagnostics.table_prior`。

LingBot 不替换已接受原始实测点，只向原始空洞增加补全点；新增点必须通过原始深度支持域、
局部深度一致性、机器人剔除及另一相机的原始实测重投影核验。深度跳变附近、
大块无支持空洞、无另一视角支持或与另一视角空闲观测冲突的补全不进入安全点云。
被拒绝的原始点不支持补全，也不能当成空洞由模型补回。
`--completion-absolute-tolerance-m` / `--completion-relative-tolerance` 默认 15 mm / 0.015。
补全深度不参与桌面先验或 unknown 清除；诊断见 `diagnostics.depth_evidence`。

## 双视角人工 OBB / 体素验证

离线任务目录的逐帧标注见 [批量 RGB 标注](../docs/BATCH_OBSTACLE_ANNOTATION.md)。
在线检测 `RealSenseD435iSource` 在读取同步帧组时执行 SDK `rs.align(color)`；
点云、机器人剔除和地图均使用 RGB 像素网格中的对齐深度，不需要再对齐一次。

启动检测后打开 `http://127.0.0.1:8766/manual-obstacles`，在 front/side RGB 上画
二维 OBB。冻结帧的 SDK 对齐深度用于将框内像素反投影到 `left_base`，随后只保留
地图中 occupied 且体素中心投影落在两路同编号框内的体素。这是双视角图像约束
下的三维体素筛选，不等于从二维框直接得到了 3D OBB；页面可选从筛选体素拟合
3D OBB。原生 Z16 深度另存为审计证据。捕获和标注元数据均写
`annotation_image=recorded_rgb`。人工筛选不会改动在线检测服务的安全地图。
执行无模型数据集避障时，可显式指定保存的 `selection.json`：

```bash
python -m obstacle_validation.lerobot_replay replay \
  --dataset outputs/obstacle_validation/trans_lerobot \
  --mode avoid --live --manual-selection outputs/manual_obstacles/<capture_id>/selection.json \
  --execute --execute-ack I_UNDERSTAND_DUAL_UR7E_WILL_MOVE \
  --output outputs/obstacle_validation/trans_avoid
```

回放每帧在保存的 3D OBB 范围内筛选实时 occupied 体素，再生成避障 OBB。
仅此显式模式把未标记区域排除出回放的体素扫掠检查；标记区域内的 unknown
仍阻止经过它的动作。回放会按 `left_base` 世界坐标把保存的 OBB 投到当前体素网格；
标定摘要不一致、标记障碍物落在当前地图工作区外，或双相机测量不足时会拒绝运行。
标记范围固定，物体移出范围后不会被筛选；
应重新捕获并标记。运行前先检查页面预览中选中体素和 3D OBB 覆盖了目标障碍物。
回放报告中的 `stale_scene_pauses` 记录因过期快照而先停臂、等待新帧并重新核验动作的次数；
此等待不延长快照的 2 秒有效期。
