# LeRobot 双臂动作回放与避障验证

## LeRobot 数据集动作回放（含双夹爪）

`obstacle_validation.lerobot_replay` 是唯一的回放验证入口。它把完整 Quest3 HDF5
转换为标准 LeRobot 目录，其中 `observation.state` 和 `action` 均为 14 维：
前 12 维是按左臂、右臂排列的关节角，末 2 维是左右夹爪闭合比例。
原始 `action[t]` 的定义是 `target_q[t] - qpos[t]`，转换时先计算同帧的
**绝对目标** `qpos[t] + action[t]`，然后按录制时间插值到默认 20 Hz。
末帧没有有效 action，使用实测末姿态作为终点。夹爪采用 HDF5 的
`gripper_action` 命令值；初始姿态和夹爪开度来自首帧实测状态。

这里的“推理”由 LeRobot 数据集逐步提供下一段 action chunk，不需要模型权重。
每个控制周期读取当前双臂和双夹爪反馈，取下一条绝对动作，计算关节目标差，
在发送前检查固定表面点与障碍物、双臂间距。`avoid` 模式对当前关节步应用
CBF 投影：用名义关节 action **和数据集的夹爪目标**计算未来固定表面点，
取它们对障碍物和另一机械臂的预测侵入距离，再在未来姿态线性化，
只优化 12 个关节增量，夹爪目标不被修改。求解后检查关节与夹爪从当前实测状态
到目标状态的组合 FK 扫掠；该组合动作不安全时停止，不会只执行关节而冻结夹爪。
路径回放时，若实时障碍框使当前固定点低于安全间距，不直接拒绝：CBF 仍基于
未来表面点修正关节动作。若关节步长不足以一步恢复完整间距，会降低本步 CBF
恢复比例；修正后的未来点必须比当前点更远离障碍物，扫掠过程也不能进一步
侵入。下一周期再用实测状态继续优化，直至恢复间距。无法得到有效改善时停止。
复位阶段仍检查起始姿态。
没有视觉障碍物的 `baseline` 仍检查双臂自碰及当前固定点几何。
实时检测仅缓存最新完整快照，局部概率占据地图默认开启，可用 `--occupancy-map`
显式开启，或 `--no-occupancy-map` 关闭。关闭时切换到过滤点云的 DBSCAN + 时序 OBB
后端（`obstacle_backend=clustered_obbs`），不发布 `occupancy_map`；保留深度噪声过滤、
悬空检测和已观测障碍丢失后仅转 unknown 的规则，unknown 仍整体阻止运动。
此模式没有体素未观测/过期空闲区域的扫掠检查，检测由点云采样、聚类点数门槛和
OBB 时序参数决定；候选框覆盖当前检测簇，连续可信原始空闲观测才能解除旧障碍的
unknown。下面的地图信念、完整像素终点和体素扫掠行为仅在地图开启时适用。
已接受原始深度终点提供
占据证据，终点前有效射线提供空闲证据，同帧占据优先；补全只增加经核验的
占据点。安全地图不再依赖 DBSCAN 簇编号、最小点数或旧 OBB 变化确认窗口。
原始深度先检查局部深度一致性与三维邻域支持：默认 3×3 邻域内至少 2 个
非机器人有效邻点，深度差不超过 8 mm + 0.005×深度、三维距离不超过 20 mm。
仅有 1 个支持邻点时需默认 3 次连续一致观测确认；超过 2 秒间隔、无效/丢失
测量或标定变化会重置历史。完全孤立点重复出现也不会自动确认，但可以由另一
相机独立接受的实测表面确认，两个未确认噪点不能相互确认。
渲染机器人后方的矛盾返回不可用于占据或空闲；机器人前方真实表面仍保留。
被拒绝深度的终点立即为 unknown，撤销历史 free 信念并否决同帧空闲射线及补全；
可信原始占据证据仍优先。未曾确认的噪点不登记为历史障碍，不造成整场景的
历史障碍 unknown 停机，但扫掠检查拒绝进入其 unknown 体素。已确认障碍被拒绝
或漏检时仍保留历史危险标记、不保留 OBB，不能借噪声过滤清空危险记录。
无需最小簇点数，细线和小物体按局部证据检查；原始值不做中值替换或空间填洞。
原始占据及可疑终点均使用完整像素证据，显示点云的 `--point-stride` 不会漏掉
采样格之间已确认的小表面；补全点仍按配置采样。
占据体素在小块内合并成实心长方体 OBB，随后应用 `--box-margin-m`，不全局合并
膨胀框。历史占据漏检时撤下几何 OBB，仅转为 unknown，不因连续遮挡、帧数或
概率衰减而清空；连续原始空闲证据达到默认 3 帧且概率降至阈值后才清除，重新
观测到终点也能恢复。历史危险 unknown 仍使回放和执行器整体停止。
`occupancy_map` 在 compact 快照中保留坐标系、采集时间、压缩状态/信念/观测年龄。
从未观测的区域及已过期的空闲区域也为 unknown；复位 RRT、FK 组合动作扫掠、
CBF 的恢复与安全前缀复核均拒绝涉及此类体素的动作。起点 unknown 不豁免，
小步移动及只动夹爪不能绕过；纯保持仍须通过原有障碍物和双臂几何检查。
默认空闲证据 TTL 2 秒，年龄包含采集、处理及控制时间，重复读取同一帧不刷新。
地图仅覆盖配置的双臂工作区并集及桌面上方，范围外不提供环境安全保证。
机械臂剔除附近未取得环境观测时可能保守拒绝运动。
旧检测器的不稳定聚类快照仍保持兼容消费；新版地图快照的历史危险 unknown
不能通过 `allow_tracked_obbs` 回退绕过。`unstable_scene_reads` 仅统计旧聚类快照回退。
桌面仍使用原始实测点建立有界先验；悬空物体在工作区内参与地图检测，默认无
0.35 m 上限。参数及诊断见 [实时感知说明](../real_scripts/README.md)。
此入口按数据集动作顺序回放；仅复位阶段在局部 CBF 无法推进时使用 RRT。
每条动作发送双夹爪命令后，按实测关节角到达绝对目标推进，默认关节到达容差
`--arrival-tolerance 0.04` rad；关节追不上时重复当前动作。`baseline` 超过
`--action-timeout 5` 秒则停止；`avoid` 不设单条动作到位超时，而是每周期重新读取
关节、夹爪和障碍物并继续计算 CBF，直到关节到位或其他运动检查失败。若目标持续
不可达，`avoid` 会停留在当前动作，需由操作员停止。夹爪命令值不是实测路点：
2F-85 的满闭合命令
`1.0` 可能在抓住物体或机械终点时停在低于 `1.0` 的读数。回放仍逐周期读取
夹爪实际开度用于固定点 FK、CBF 和组合运动检查，并记录与命令值的误差；
`--gripper-tolerance` 只用于确认初始姿态复位。

先离线转换和验证 `episode_00003`：

```bash
python -m obstacle_validation.lerobot_replay convert \
  --hdf5 ../quest3_collect/data/dual_vr_episodes/pick_the_centrifuge_tube/episode_00000.hdf5 \
  --output outputs/obstacle_validation/episode_00000_lerobot

python -m obstacle_validation.lerobot_replay replay \
  --dataset outputs/obstacle_validation/episode_00000_lerobot \
  --mode baseline \
  --output outputs/obstacle_validation/episode_00000_virtual
```

关闭遥操作程序，确认机械臂控制权已交给此进程。`baseline` 实机回放先用首帧
实测状态对关节和双夹爪作平滑插值复位。`avoid` 在复位阶段也使用固定点 FK、
实时 OBB 和双臂间距检查：每周期生成朝录制起点的有界关节 action，先预测该
action 的未来双臂表面点，再由同一 QP-CBF 修正可能侵入的动作。只有局部 CBF
无法推进、且录制起点本身安全时，才尝试 FK 检查的局部 RRT 绕行。运动时保持
当前夹爪开度，到达安全姿态后逐步恢复双夹爪。若当前开度下
录制起点不安全、目标开度下安全，会先在当前位置安全地调整夹爪。每步发送前
重新读取障碍物快照；当前姿态或录制起点被障碍物占据、绕行失败、夹爪动作受阻
或复位超时时停止，`report.json` 的 `failure_context` 保存当步关节、夹爪、OBB
和失败原因。`restore_cbf_interventions` 记录复位 action 被 QP 修正的次数，
`restore_replans` 记录局部 CBF 不能推进后启用 RRT 的次数。默认总复位超时为 90 秒，
可用 `--restore-timeout`、`--restore-planning-timeout` 和 `--restore-detour-room`
调整复位时间及关节绕行范围。等待反馈到达录制起点后，才从数据集第一个
action 开始按 20 Hz 发送。实机输出仍要求明确的执行参数：

```bash
python -m obstacle_validation.lerobot_replay replay \
  --dataset outputs/obstacle_validation/episode_00000_lerobot \
  --mode baseline --execute \
  --execute-ack I_UNDERSTAND_DUAL_UR7E_WILL_MOVE \
  --output outputs/obstacle_validation/episode_00000_baseline

# 启动实时检测服务并布置障碍物后，用同一数据集验证 CBF。
python -m obstacle_validation.lerobot_replay replay \
  --dataset outputs/obstacle_validation/episode_00000_lerobot \
  --mode avoid --live --execute \
  --execute-ack I_UNDERSTAND_DUAL_UR7E_WILL_MOVE \
  --output outputs/obstacle_validation/episode_00000_avoid
```

也可把 `--dataset` 换成 `--hdf5`，首次自动在输出目录生成 LeRobot 数据集。
每次运行产生 `steps.jsonl` 和 `report.json`，记录实际发送的关节增量、
夹爪目标、数据集动作、CBF 结果、实测追踪误差、实际路径耗时和每步计算耗时。
若几何检查拒绝发送，`report.json` 的 `failure_context` 记录失败类别、当前及
目标间距、双臂开度、CBF 输出和当步 OBB，便于判断是关节、夹爪还是组合动作受阻。
CBF 求解失败也会记录当前关节、数据集目标、名义关节步、上一关节步、关节边界、
实时 OBB 和当前间距。令 `d_nom` 为候选 action 关节步，`d` 为修正步，CBF 对
未来固定点施加 `h_future + J_future(d-d_nom) >= (1-alpha)h_current`；
`h_future` 是未来点到 OBB 的欧氏带符号距离减去安全间距，双臂点对同理。
这样预测侵入直接决定 action 修正量；当前距离用于核对脱离过程是否继续侵入。
若为了避障必须更快制动，求解器可放宽名义加速度边界，但仍检查关节速度、
关节范围和整段 FK。未来点线性化因跨越 OBB 面而无解时，只允许使用经 FK 验证的
原动作安全前缀；线性化结果越界时也会沿该步回退到经几何验证的安全位置。
轨迹边检查使用与 CBF
一致的 OBB 欧氏距离边界；对角落附近的安全点不会仅因逐轴扩大的保守包围盒而
误判。若回放中当前点低于安全间距，CBF 尝试使未来点逐步脱离；若未来点
无法改善或运动边进一步侵入，则停止并在 `failure_context` 中留下诊断。
实机出错时还会尽量用最新实测开度命令双夹爪
停在当前位置，并在 `gripper_hold_on_error` 中记录结果。
复位及路径回放均以当前实测状态计算关节目标；前一条 servoJ/夹爪命令还在运动时，
下一周期的反馈变化属于正常现象。`--live` 在发送前还会用更新的障碍物快照
重新计算并检查该步。
`episode_00002` 默认录制时长约 14.47 秒，转换为 290 条 20 Hz 动作；
左臂基本静止，右臂最大单关节范围约 0.61 rad。实际耗时会受反馈、几何计算、
夹爪速度和 CBF 影响；如果跟踪误差过大，回放会停止而非默默跳过动作。
当机械臂无法跟上 20 Hz 数据集节奏时，路径实际耗时会超过 14.5 秒；
`report.json` 分别记录已完成动作数、重复当前动作的控制周期数和总控制周期数。
该文件的左夹爪首帧实测闭合比例约 0.553，而首条 `gripper_action` 为 0；
因此复位先恢复 0.553，路径开始后会按录制命令打开左夹爪。
右夹爪在源帧约 93 处闭合、约 204 处重新打开。
Robotiq 写入沿用采集端的一次性 `SET POS/SPE/FOR/GTO` 命令；该 URCap
不保证对每条 `SET` 分别回执。复位是否完成由独立的只读 `GET POS` 实测反馈
确认，超过 `--restore-settle-timeout`（默认 10 秒）仍未到达时停止。


本验证用已记录动作检验固定表面点几何和避障闭环，不评估学习模型质量。
