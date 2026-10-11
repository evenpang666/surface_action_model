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
实时检测只缓存最新完整快照；每个 OBB 的点簇观察最多保留最近 10 帧。
点集稳定时固定盒子几何，默认比较当前帧和前两帧，三帧持续显著变化才更新。
漏检的旧框最多保留 10 帧；若相机深度连续 3 帧测到旧点簇表面后方的
空闲空间，则提前移除。连续遮挡 10 帧后也会删除旧框。
`--obb-history-frames` 可调整窗口长度。重复轨迹会合并。新簇、
待确认的漏检或多个不同簇突然合并会令快照状态变为
`error: unstable obstacle point clusters`。安全快照包含所有已确认旧框，以及旧框
未覆盖的当前候选框。两个框的交集超过较小框体积的 50% 时，发布前优先合并，
高于稳定框保持原形的规则；合并框包住原框。实机回放直接使用这些 OBB 继续 CBF 和扫掠检查，不等待
点簇状态恢复。旧版检测器的 `merged_clusters`、`missing_cluster` 也可直接使用
已发布的旧框；`new_cluster`、`changed_cluster` 需新版检测器发布候选框。
报告中的 `unstable_scene_reads` 记录使用此类快照的次数。
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
