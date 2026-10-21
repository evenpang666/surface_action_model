# 回放避障

实时模型输入可使用 [oavla socket＋共享相机避障入口](OA_VLA_SOCKET_RUNTIME.md)。
该入口直接消费模型配对 OBB，后台建图，无需人工选择文件；下文是录制动作回放入口。

无需模型权重。使用双 UR7e、双 Robotiq 和已标定的 front/side 相机；
先安装[LeRobot、OSQP、Voxblox](IMPLEMENTATION_GUIDE.md#环境)，在 `surface-runtime` 环境运行。

## LeRobot 入口：直接使用 RGB annotations.json

已有数据集和标注时直接执行：

```bash
python -m obstacle_validation.lerobot_replay replay \
  --dataset outputs/obstacle_validation/my_episode_lerobot --mode avoid
```

**默认实机执行。** avoid 自动启动相机与 Voxblox，并从 `outputs/manual_obstacles/`
下包含 `annotations.json`、目录修改时间最新的捕获目录读取标注。
输出默认在 `outputs/obstacle_validation/annotation_replay_YYYYMMDD_HHMMSS_微秒±时区/`。

| 需求 | 参数 |
| --- | --- |
| 虚拟回放，不连接运动输出 | `--no-execute`；avoid 仍采集相机 |
| 原动作直接回放，不启动感知、筛选或避障 | `--mode baseline` |
| 指定标注 | `--manual-annotations /path/to/annotations.json`；别名 `--annotations` |
| 指定输出目录 | `--output outputs/obstacle_validation/my_run` |
| 限制回放关节速度 | `--speed 0.5`，rad/s |
| 开启机器人深度剔除 | `--robot-mask` |
| 使用外部检测进程 | `--obstacle-source external`；可配 `--obstacle-url URL` |
| 拒绝过期地图 | `--obstacle-age-policy strict`；默认 latest 使用最新缓存 |

baseline 不接受标注/筛选参数。
复位阶段按录制初始关节和夹爪状态插值，**不执行避障或碰撞检查**；到位后开始数据集动作。
avoid 允许距离约束松弛；夹爪保持录制目标。

## 准备数据与标注

已有 LeRobot 数据集可跳过转换：

```bash
python -m obstacle_validation.lerobot_replay convert \
  --hdf5 /path/to/episode.hdf5 --output outputs/obstacle_validation/my_episode_lerobot
```

采集并保存 RGB 旋转框：

```bash
python real_scripts/manual_obstacle_annotation.py
```

1. 退出其他占用相机的进程，打开 <http://127.0.0.1:8770>。
2. 点击捕获，在 front/side 中为同一物体使用相同编号画框、旋转并保存。
3. 关闭标注页或确认捕获已释放相机，再运行回放。

同一物体必须同时落入两个视角的对应框；多个物体取并集。
筛选使用实时体素，不读取旧 `selection.json` 或旧地图。单臂标注不能用于双臂回放。
旧 `annotations` 格式需保留同目录的 `capture.json`；新 `boxes` 格式自带 RGB 内参。

### 双视角确认、无体素时的透明障碍

默认逐对象检查两个对应框内的占据体素。若某对象没有占据体素，仍保留它作为透明障碍：
将 front/side 旋转框按标定反投影为视锥，在工作空间网格的有向边界及两相机 0.01–2.5 m 光轴深度内求交，
为交集直接生成一个包围椭球，供回放的 MCBF/QP 优化使用。有体素的对象继续采用实测几何；
后续帧出现体素时自动切回实测。两条链路共用透明区域建模。模型推理会忽略无交集的低置信配对并保留实测体素约束；人工标注回放仍严格校验配对。

补全不添加伪造的深度点或占据体素。日志和报告的 `paired_geometry` 区分对象的实测体素数与透明补全；
快照 `obbs` 中补全项带 `geometry_source=dual_rgb_transparent_region` 和 `ellipsoid_shape`。
即使物体小于体素分辨率，也可用连续几何生成区域；双视角没有有效空间交集或缺少一侧框时拒绝处理。

该区域表示基于 RGB 的保守障碍假设，无法证明物体材质或真实厚度。椭球可能大于实物，
工作空间边界也可能跨越双臂区域之间的间隙；畸变边界使用采样与余量近似。
具体字段见 [透明障碍补全](OA_VLA_SOCKET_RUNTIME.md#无体素区域的透明障碍补全)。

## 双终端 action stream（可选）

发送端，默认每块 32 条动作，监听 `127.0.0.1:8877`：

```bash
python -m obstacle_validation.action_stream \
  --dataset outputs/obstacle_validation/my_episode_lerobot
```

接收端需要旧式 `selection.json`，可在[实时标注页](../real_scripts/README.md#双视角人工-obb--体素验证)生成：

```bash
python -m obstacle_validation.stream_replay \
  --manual-selection outputs/manual_obstacles/CAPTURE_ID/selection.json \
  --output outputs/obstacle_validation/stream_run
```

接收端**默认虚拟执行**。实机加 `--execute --execute-ack I_UNDERSTAND_DUAL_UR7E_WILL_MOVE`；
去掉 `--manual-selection` 为 baseline。发送端等当前块执行确认后才发送下一块。
跨机时两端设置一致的 `--host` / `--port`；断线后不会自动重放。

## 参数和日志

| 参数 | 默认 / 作用 |
| --- | --- |
| `--restore-speed` | 0.5 rad/s，复位关节速度 |
| `--gripper-speed-fraction-s` | 0.5/s，复位夹爪速率 |
| `--restore-timeout` | 90 s，复位总超时 |
| `--gripper-speed` | 255，夹爪硬件速度；不改变录制目标 |
| `--acceleration` | 可选，回放加速度限制 rad/s² |

控制器配置：`configs/ur7e_dual_quest3.yaml` 的 `safety.ellipsoid_mcbf`。
调节 `obstacle_max_parts` / `obstacle_max_span_m` 改变障碍拟合精细度；
`collision_slack_weight` / `stagnation_steps` 调节松弛与停滞处理。
约束包括左右臂各自对障碍、双臂之间；不包含同一臂内的避障。

| 文件 | 查看内容 |
| --- | --- |
| `report.json` | 完成状态、执行动作数、错误与耗时 |
| `steps.jsonl` | `phase=restore/path`、反馈、控制目标、松弛量 |
| `chunks.jsonl` | 动作块与规划/执行记录 |

默认不加载 LingBot，关闭 robot mask；内嵌回放关闭 ESDF 和网页。
需要看地图时单独启动[障碍可视化](../real_scripts/README.md)，并通过外部检测模式复用该进程，避免相机冲突。
