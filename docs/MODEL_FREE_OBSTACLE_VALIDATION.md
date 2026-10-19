# LeRobot 双臂动作回放与避障验证

## LeRobot 数据集动作回放（含双夹爪）

`obstacle_validation.lerobot_replay` 是唯一的回放验证入口。它把完整 Quest3 HDF5
转换为标准 LeRobot 目录，其中 `observation.state` 和 `action` 均为 14 维：
前 12 维是按左臂、右臂排列的关节角，末 2 维是左右夹爪闭合比例。
原始 `action[t]` 的定义是 `target_q[t] - qpos[t]`，转换时先计算同帧的
**绝对目标** `qpos[t] + action[t]`，然后按录制时间插值到默认 20 Hz。
末帧没有有效 action，使用实测末姿态作为终点。夹爪采用 HDF5 的
`gripper_action` 命令值；初始姿态和夹爪开度来自首帧实测状态。

这里的“推理”由 LeRobot 数据集提供 action chunk，不需要模型权重。`avoid` 每次
读取预测 chunk（`--action-horizon`，默认 32），只优化和执行它的前
`--execute-horizon` 条（默认 10），对应 VLA 的 `action_chunk[:10]`。
执行完这 10 条后，从数据集下一条重新读取预测 chunk；后 22 条不参与本轮 FK、QP
或扫掠规划。设置 `--execute-horizon 5` 可改为只处理前 5 条。实际处理长度取
预测长度、执行长度和数据集剩余数量的最小值，末尾不足时消费剩余目标。

执行前缀采用**逐步 CBF-QP 优化**：从当前实测关节和夹爪开度出发，预测各 action 对应
的未来固定表面点。某一步接近所选 occupied 体素的安全边界或双臂过近时，在该步
自己的预测姿态和雅可比上求解 12 维关节 QP；随后用修正后的关节姿态展开后续目标。
因此执行前缀后半段的多个风险 action 都会在开始执行这段前缀前得到优化。这是按时间
顺序的多个小 QP，不是把全部关节目标放入一个联合 QP。夹爪命令值保持数据集目标，
其几何变化纳入各步 FK 及扫掠；无风险动作保留原目标。原目标距离超过单周期速度
上限时，展开为多个控制路点，不因步长限制而跳过原目标。

阈值包含 `collision_margin_m` 和 `trigger_margin_m`，在实际进入体素之前触发。
障碍几何为实时 occupied 体素的实心分块 OBB，保留检测器的 `box_margin_m` 膨胀，
不把整个人工选框当作障碍实心体。每次 QP 后都检查关节和夹爪的组合扫掠；未来法向
在圆角或障碍内部不能给出安全修正时，使用当前表面法向、雅可比和夹爪形变重新求解，
并复核扫掠。QP 使用 10 μm 数值裕量，安全的零关节步只有在夹爪扫掠也安全时接受。
执行前缀内任意一项优化失败会使本段计划失败，不发送尚未完成验证的前缀。

实机在构造执行前缀的计划前先停臂并读取反馈，使整段优化的起始姿态稳定。规划使用当时
的地图分类；执行每个控制路点前，仍读取实测关节、夹爪及后台缓存中的最新地图，
检查实际即将发送的扫掠。新障碍阻挡或反馈漂移时停臂，基于最新状态重新优化当前
执行前缀尚未执行的目标。默认 `latest` 模式不因采集时间变旧而停臂或等待新帧。
安全目标按顺序消费；到达的是**优化后的目标**，不再等待被障碍挡住的原目标。
完成后报告原始终点误差 `final_joint_error_rad` 和优化终点误差
`final_optimized_joint_error_rad`，`avoid` 完成条件使用后者。

`chunks.jsonl` 记录每次预测长度 `inferred_horizon`、本次规划长度 `planned_horizon`，
以及执行前缀的原始目标、优化目标、控制路点、路点所属 action，
以及触发 QP 和被修改的 action 索引。`steps.jsonl` 保留原始 `action`，新增
`optimized_action`、`control_target`、`chunk_id`、`chunk_step` 和
`original_target_error_rad`；`target_error_rad` 表示距优化目标的误差。每步的 `cbf`
记录其规划时的求解结果，复用该路点时不会因此再次求解。`report.json` 中
`cbf_chunk_mode=optimize_execution_prefix`，`action_horizon` 和 `execute_horizon`
分别记录预测长度与执行前缀上限，`chunk_plans` 和 `chunk_replans` 分别记录
实际计划数量和安全检查导致的计划失效次数。
路径回放时，若实时障碍框使当前固定点低于安全间距，不直接拒绝：CBF 仍基于
未来表面点修正关节动作。若关节步长不足以一步恢复完整间距，会降低本步 CBF
恢复比例；修正后的未来点必须比当前点更远离障碍物，扫掠过程也不能进一步
侵入。后续周期用实测状态核验恢复路点，原队列不能恢复时重新优化剩余目标；无法得到有效改善时停止。
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
观测到终点也能恢复。历史危险 unknown 仍使默认回放和执行器整体停止。
若使用 `lerobot_replay replay --mode avoid --live --manual-selection selection.json`，
回放按人工保存的 3D OBB
范围筛选每帧实时 occupied 体素并生成避障框。此显式人工验证模式只对标记范围
执行体素扫掠检查，因而未标记区域的历史 unknown 不再整体停止回放；范围内的
unknown 仍会阻止扫掠。保存的 3D OBB 会按 `left_base` 坐标重新投到当前网格；
标定不匹配、标记范围落在地图工作区外或相机工作区测量不足时仍拒绝运行。
`occupancy_map` 在 compact 快照中保留坐标系、采集时间、压缩状态/信念/观测年龄。
检测器发布的 unknown 区域仍会阻挡复位 CBF-QP、组合动作扫掠和恢复动作。
起点 unknown 不豁免，小步移动及只动夹爪不能绕过；纯保持仍须通过障碍物和双臂
几何检查。体素邻域用于粗筛，最终按 FK 点运动线段到体素的最小欧氏距离检查，
包括圆角和地图边界；10 mm 间距不会因 20 mm 分辨率扩大为整圈相邻体素。

避障回放默认 `--obstacle-age-policy latest`：没有新帧时持续复用缓存中最新帧，
不因超过原来的 2 秒时间限制而停臂等刷新。对该帧 free/occupied/unknown 的判断
使用它在采集时发布的占据信念，不在控制端随经过时间继续衰减 free 证据。
原始采集时间、单格观测年龄和 TTL 不改写；新帧到达后立即采用新分类。
**这会取消检测停止更新后的时间失效保护：检测器停止或网络超时期间，机械臂仍可
依据旧场景继续运动，新出现的障碍在新帧到达前不会被获知。**
unknown、occupied、无效格式、标定错误及机器人反馈异常仍拒绝执行，网络超时不会
清除已经收到的 unknown/无效状态。

需要原有时效行为时使用 `--obstacle-age-policy strict`：按实际采集年龄检查快照和
free 证据，启动最多等待 5 秒取得有效帧；人工标记实机避障遇过期时先停臂，最多等
3 秒更新采集帧后重新规划。默认 free TTL 2 秒，重复读取同一帧不刷新时间。
`latest` 模式启动仍需取得一张结构及标定有效的快照，HTTP 请求保持有限超时。
地图仅覆盖配置的双臂工作区并集及桌面上方，范围外不提供环境安全保证。
机械臂剔除附近未取得环境观测时可能保守拒绝运动。
旧检测器的不稳定聚类快照仍保持兼容消费；新版地图快照的历史危险 unknown
不能通过 `allow_tracked_obbs` 回退绕过。`unstable_scene_reads` 仅统计旧聚类快照回退。
桌面仍使用原始实测点建立有界先验；悬空物体在工作区内参与地图检测，默认无
0.35 m 上限。参数及诊断见 [实时感知说明](../real_scripts/README.md)。
此入口的复位及数据集动作回放都使用 CBF-QP，不调用 RRT。
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
action 的未来双臂表面点，再由同一 QP-CBF 修正可能侵入的动作。通过实际扫掠检查的
侧向或暂时远离目标的 QP 修正也可执行，不要求每步沿目标方向前进。QP 给出安全
零步时停臂保持并在下一周期重新求解，受总复位超时约束；不回退到其他路径规划器。
运动时保持
当前夹爪开度，到达安全姿态后逐步恢复双夹爪。若当前开度下
录制起点不安全、目标开度下安全，会先在当前位置安全地调整夹爪。每步发送前
重新读取障碍物快照；当前姿态或录制起点被障碍物占据、QP 不可行、夹爪动作受阻
或复位超时时停止，`report.json` 的 `failure_context` 保存当步关节、夹爪、OBB
和失败原因；失败时还记录 `cbf_step_duration_s` 及扫掠拒绝的原因。
`planning_observation_rejections` 区分 unknown、过期 free 和 occupied，
`planning_observation_blockage` 记录最后阻挡的体素及 FK 点编号；同时保存筛选后的
`occupancy_map` 和 `planning_observation_check_ns`，供离线重放同一地图及观测时刻。
`restore_cbf_interventions` 记录复位 action 被 QP 修正的次数，
`restore_controller=cbf_qp` 表示复位仅使用 QP；兼容字段 `restore_replans` 恒为 0。
`restore_cbf_hold_ticks` 记录安全保持的周期数。
默认总复位超时为 90 秒，可用 `--restore-timeout` 和 `--restore-joint-room`
调整复位时间及关节约束范围。默认 `latest` 模式在复位和路径阶段均复用最后一帧，
不等待时间更新。`strict` 模式遇到过期 free 证据才停臂等待更晚的采集帧，再求解 QP。
等待反馈到达录制起点后，才从数据集第一个
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

实时地图由后台线程获取，完成启动取图后，控制周期直接读取缓存的最新快照，
不会为每个 action 等待检测器刷新，也不会在发送前同步请求 HTTP。
后台每次请求完成后间隔 100 ms 再拉取；新地图到达后替换缓存，同一张地图可以供
多个 action 使用。发送前仍读取当前机械臂反馈，并用此时缓存中的最新地图检查扫掠。
同一快照的体素只解码一次。默认 `latest` 不执行时间过期等帧；
`strict` 仍按原采集时间和 free 证据有效期检查。

每次规划只检查执行前缀的 FK 和扫掠，只对其中有风险的步求解 QP；队列执行时
只检查即将发送的控制路点。默认障碍物触发距离是距筛选体素生成的 OBB
`collision_margin_m + trigger_margin_m = 35 mm`，不是等到点进入体素。
规划中完全相同的关节及夹爪姿态复用 FK；实际发送前仍用实测状态检查扫掠。

排查卡顿时查看以下报告字段（计时包含失败及重试调用，单位秒）：

| 字段 | 含义 |
| --- | --- |
| `runtime_timings` | 各阶段的 `calls`、`total_s`、`max_s`，包括执行前缀 CBF 规划、反馈、发送前扫掠、关节/夹爪命令、停臂及等新帧 |
| `snapshot_timings` | 后台 HTTP/JSON、人工筛选、校验解码和控制线程首次解码的耗时 |
| `snapshot_cache` | 控制取缓存次数、后台刷新次数、网络错误次数及最后错误和后台拉取间隔 |
| `obstacle_age_policy` / `stale_snapshot_reads` / `max_snapshot_age_s` | 时效策略 / 超过原时间阈值的缓存读取次数 / 最大实际采集年龄 |
| `cbf_runtime` | 全部投影尝试数、实际 QP 求解次数和约束筛选、雅可比、QP、扫掠等阶段累计耗时 |
| `planning_fk_cache` | 执行前缀规划 FK（含有限差分雅可比采样）的缓存命中及计算次数 |
| `watchdog_stops` / `stale_scene_pauses` | 命令看门狗停臂次数 / 感知证据过期停臂等待次数 |

`steps.jsonl` 每步的 `cbf.qp_solve_calls` 区分该路点规划时是否求解，`cbf.timing_s`
记录该控制路点规划时的 CBF 分项耗时，复用路点时不会新增求解；实际求解总数以 `cbf_runtime` 为准。`runtime_timings.chunk_cbf_pipeline` 已包含 `cbf_runtime`
的子阶段，`scene_and_feedback` 也包含反馈及解码，不能把父子计时重复相加。
`mean_cycle_compute_s` 只统计成功写入步骤日志的周期，排查重试或等待需结合上述累计计时。

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
