# 实现与运行细节

本文保存项目的数据、模型、标定和在线运行细节。入门流程见[项目 README](../README.md)。

基于 OpenPI PI0.5 的双臂 action / surface 联合预测。方法类为
`SurfaceActionModelPytorch`；保留上游 `PI0Pytorch`、`pi05_base` 名称。

```text
Quest3 HDF5 → FK 可见性筛选 → CoTracker + 实测深度 → surface/action shard
RGB + task + state → 冻结 PI0.5 VLM → action expert（训练、flow matching）
VLM 输出 prefix + current_surface + action_chunk → point DiT → point decoder
    → flow velocity → Euler 积分 → surface_chunk
front/side D455 → 原始深度与核验补全 → 双臂剔除 → 概率占据地图 → 分块 OBB
    → 未来 surface 检查 → 首动作关节 CBF-QP → FK 轨迹复核
```

点分支以同一次生成的 action chunk 为额外条件，使随机点预测与动作对应。
训练只冻结 VLM；action expert、动作投影和 point DiT/decoder 都参与训练。

## 当前硬件与标定

左右工具均为 Robotiq 2F-85。只读枚举确认：

下列序列号来自硬件配置；本轮标定接入仅验证文件和几何计算，未验证设备在线状态。

| 角色 | 型号 | 序列号 |
| --- | --- | --- |
| front | D455 | 260922305390 |
| side | D455 | 260922303231 |
| left_wrist | D435i | 420222071502 |
| right_wrist | D435i | 405622072940 |

已接入 **2026-10-08 双臂双视角标定**：
`../quest3_collect/config/calibration/dual_arm_camera_transforms.yaml`。
障碍物检测与固定表面点 FK 共用汇总文件中的基座/相机变换，并验证四份
源文件的 SHA256、质量与坐标闭环。快照标定哈希不一致时拒绝执行。
流程见 [CALIBRATION.md](CALIBRATION.md)。历史外参仅适用于对应旧采集数据。

## 环境

Python 3.11。策略与感知分环境，避免 OpenPI 与 LingBot 原始 torch 要求互相覆盖。
在仓库根目录执行以下命令；requirements 内 editable 路径按其目录解释：

```bash
conda create -n surface-policy python=3.11 -y
conda run -n surface-policy bash -c 'cd requirements && python -m pip install -r policy.txt'
conda create -n surface-runtime python=3.11 -y
conda run -n surface-runtime bash -c 'cd requirements && python -m pip install -r runtime.txt'
conda run -n surface-runtime python -m pip install --no-deps -e ./lingbot-depth
```

预处理使用 `requirements/preprocess.txt`；CoTracker 源码/权重通过
`--repo`、`--checkpoint` 指定。RealSense USB 权限/驱动和 UR 网络为系统依赖。
ChArUco 需要同版本 opencv-contrib-python **替代**普通 OpenCV wheel，不要混装。

清单固定直接依赖，不是已验证的全量传递依赖锁；本轮测试使用本机现有
`conda safety`，没有改变它的软件安装。新环境仍需单独安装验证。
OpenPI 自带 uv.lock 可用于其独立环境。

Transformers 固定 4.53.2；扩展由 `transformers_runtime.py` 从仓内按进程加载，
**不再覆盖 site-packages**。PI05 基础权重仍通过 OpenPI 的
`examples/convert_jax_model_to_pytorch.py` 从 pi05_base 转换。

## 数据预处理

从仓库根目录运行。必须使用该 episode 采集时有效的标定。
`--geometry-only` 的产物只用于 CoTracker 可见性筛选。

```bash
python scripts/preprocess_quest3_hdf5.py \
  --episode ../quest3_collect/data/dual_vr_episodes/episode_00000.hdf5 \
  --output outputs/gates/episode_00000.shard --geometry-only

python scripts/preprocess_cotracker_surface.py \
  --episode ../quest3_collect/data/dual_vr_episodes/episode_00000.hdf5 \
  --geometry-shard outputs/gates/episode_00000.shard \
  --calibration ../quest3_collect/config/calibration/left_base_to_front_camera.yaml \
  --output outputs/tracks/episode_00000.npz \
  --preview outputs/tracks/episode_00000.png \
  --visualization outputs/tracks/episode_00000.html

python scripts/preprocess_quest3_hdf5.py \
  --episode ../quest3_collect/data/dual_vr_episodes/episode_00000.hdf5 \
  --measured-flow-npz outputs/tracks/episode_00000.npz \
  --output outputs/surface_action/episode_00000.shard
```

CoTracker 使用双臂网格筛选实测像素；初始帧必须有两臂种子，丢失的深度/
跟踪标签由 mask 排除。目前 extractor 使用录制 front 视角，不为不可见部位
伪造监督。在线 current_surface 同样由实测深度获得，通过 link 关联计算 Jacobian。

HDF5 中的 front 深度已与录制 RGB 对齐；CoTracker 直接读取并换算米制深度，
不再执行离线 depth→RGB 重投影。`--calibration` 仍用于 front color optical frame
到 `left_base` 的变换。预处理会拒绝元数据标为原生或坐标来源不明的深度；
数据契约与单位优先级见 [Quest3 已对齐深度](QUEST3_DEPTH_ALIGNMENT.md)。

这里的 color 指固定 front 相机的 color optical frame。`left_base` 是双臂
三维点和障碍物共用的参考坐标系；右臂点先经 right_base→left_base 变换。
目前两个腕部相机仅录 RGB，不参与此 extractor 的深度监督。若以后使用腕部
RGB-D，深度→RGB 对齐仍使用各相机自身标定，而 RGB→left_base 必须逐帧结合
所属机械臂 FK 和 flange→camera 手眼标定，不能套用固定 front 相机的外参。

默认 `.shard` 是可 mmap 的 NPY 数组目录，也支持旧 NPZ。旁边保存原始 HDF5
完整 `.source.hdf5` 副本及相对引用。每帧提供 H 步窗口；末尾 padding 有
`action_step_mask` 和点 mask，训练排除无效监督。
`--complete-windows-only` 保留旧窗口行为。
四视角 RGB、CoTracker 测量点与后续 RGB OBB 标注的 LeRobot 导出命令见
[四视角数据导出](QUEST3_DEPTH_ALIGNMENT.md#四视角-lerobot-训练数据导出)。

`current_surface[N,K,3]` 为当前实测点；`surface_chunk[N,H,K,3]` 为未来序列，
对应 `action_chunk[N,H,7|14]`；坐标统一 left_base，单位米。
点 ID 在 chunk 内一致，跨 episode 不假定 tracker 与网格点索引相同。
保留 commanded/executed 动作及 `action_dt_s`。固定周期训练拒绝不规则时间间隔，
需要先重采样，避免静默混用不同动作时长。

## 训练与策略服务

```bash
python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/surface_action.pt --validate-only

python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/surface_action --output outputs/models/surface_action.pt \
  --pretrained outputs/pretrained/pi05_base_pytorch/model.safetensors \
  --max-points 128 --batch-size 1 --precision bfloat16

python scripts/serve_quest3_pi05_safety.py \
  --checkpoint outputs/models/surface_action.pt --device cuda --port 8000
```

服务部署于 **192.168.124.64**。默认策略 RGB 输入为 front；两路 wrist 在硬件清单
中，但不是默认训练视角。感知 front/side 与策略 RGB 视角是不同契约；增加策略
视角时必须同时配置训练和部署输入。

新 checkpoint 为 `SurfaceActionModelPytorch` / schema 3。旧
`PI05SafetyPytorch` 仅可历史回放，不能通过改名字变为已训练 DiT，必须重新训练。
默认 `--point-target visual`；fixed 仅用于旧数据检查/对照。
DDP 不分片；不能依据旧 README 认定 16GB 卡可容纳 action expert 的优化器状态。
本轮未验证 V100/A100/H100 的实际显存上限。

## 在线运行

完成新标定并验证后，在测试端运行：

```bash
python real_scripts/live_obstacle_detection.py --no-browser
python real_scripts/live_obstacle_detection_viewer.py
python real_scripts/run_ur7e_vla_safety_executor.py --prompt "manipulate the tabletop instruments safely" --once
```

人工双视角 OBB 验证：检测服务启动后打开 `http://127.0.0.1:8766/manual-obstacles`，
捕获 front/side RGB-D，在两张 RGB 图上拖框、旋转；SDK 对齐深度用于三维映射，同编号体素
必须同时满足两个视角的框约束，筛选后在原 3D 视图中显示黄色体素和蓝色
3D 包围框。结果固定在捕获帧并保存到 `outputs/manual_obstacles/`，仅用于验证。
操作及两种筛选方式见 [实时感知说明](../real_scripts/README.md#双视角人工-obb--体素验证)。
任务目录下全部 HDF5 的 front/side RGB 逐帧标注与写回，见
[批量标注说明](BATCH_OBSTACLE_ANNOTATION.md)；独立脚本为 `real_scripts/batch_obstacle_annotation.py`。

默认 dry-run。快照为 `http://127.0.0.1:8766/snapshot.json`，包含双视角 OBB、
双臂 qpos、双夹爪实测状态与 current_surface。使用实际流内参；原始深度先经过
局部深度一致性、三维邻域支持与机器人遮挡检查，接受的实测值不被模型改写。
弱支持点需要连续观测或另一视角确认；孤立点不生成 OBB，也不能用于清除障碍。
地图开启时，可疑终点保持 unknown，撤销先前的 free 信念。LingBot 只增加经支持域、局部深度和另一相机
已接受原始观测核验的空洞点，补全不用于清除障碍。
地图开启时，原始占据和可疑终点均使用完整像素证据，不受显示点云的 `--point-stride` 影响。
桌面采用原始深度建立的有界先验；检测包含悬空簇，默认高度范围由工作区决定。
参数及证据诊断见 [实时感知说明](../real_scripts/README.md)。
可视化页的“隐藏机械臂（R）”按钮可切换红橙色机械臂点的显示；障碍物检测和快照数据仍照常更新。

LeRobot 动作避障回放建议使用
`python real_scripts/live_obstacle_detection.py --no-browser --no-lingbot-depth`，
避免深度修复延迟导致超过当前配置的 `stale_timeout_s: 2.0`。
验证程序自动请求 `snapshot.json?compact=1`，省去展示点云、表面点和 JPEG；
检测线程先发布安全数据，展示点列表仅在页面或完整快照请求时生成；JPEG 仅在
完整快照请求时生成。页面请求 `viewer=1` 不生成 JPEG 或表面点。默认每帧最多展示 12000 个
环境点，可用 `--display-max-points` 调整；该参数不改变检测使用的点云。
启动及执行时均直接读取当前快照；过期快照立即拒绝。
局部概率占据地图默认开启，可用 `--occupancy-map` 显式开启，用 `--no-occupancy-map`
关闭并切换到过滤点云的 DBSCAN + 时序 OBB 后端：

```bash
python real_scripts/live_obstacle_detection.py --no-browser --no-lingbot-depth --no-occupancy-map
```

关闭时不创建体素地图、不发布 `occupancy_map`，也不启用体素 unknown/过期空闲的
扫掠检查；深度噪声过滤、悬空物体检测及已观测障碍丢失后仅转 unknown 的规则仍保留。
此时 `--point-stride`、`--cluster-eps-m`、`--cluster-min-samples`、`--min-cluster-points`
和 OBB 时序参数控制检测，过大的采样步长或簇点数门槛可能漏掉细小物体。
快照 `obstacle_backend` 为 `clustered_obbs`；默认地图后端为 `probabilistic_voxels`。
以下地图行为和参数仅在开启时适用：已接受原始深度终点更新占据，终点前有效射线
更新空闲；无效深度、终点后方及机器人遮挡后方保持未知。每个双视角帧内体素
只更新一次，占据优先。地图不依赖聚类 ID 或最小簇点数；实心体素在小块内生成
OBB，不全局合并膨胀框，保留物体间隙。
历史占据失去观测即转为 unknown，不发布历史 OBB，也不会随帧数自动清空；
原始连续空闲证据和概率阈值共同决定清除。历史危险 unknown 仍整体阻止运动。
未观测或已过期的空闲体素由 RRT/FK/关节与夹爪组合扫掠检查拒绝通过；起点
unknown 不豁免。地图的覆盖范围仍是配置的工作区并集及桌面上方，范围外不提供环境保证。
`occupancy_map` 包含压缩状态、概率信念、观测年龄和来源，compact 请求也会携带。
默认体素 20 mm、空闲射线步长 6、空闲证据 TTL 2 秒；需结合实测发布延迟调参，
重复读取不会刷新观测年龄。完整参数和诊断见 [实时感知说明](../real_scripts/README.md)。
过期快照、相机或校准故障仍会报错退出。
快照诊断包含 `processing_seconds` 和 `capture_to_publish_seconds`，有效期始终从采集时刻计算。
排查检测帧率时，读取 `snapshot.json?compact=1` 的 `diagnostics`：
`capture_seconds`、`robot_geometry_seconds`、`depth_stage_seconds`、
`point_cloud_seconds`、`scene_analysis_seconds`、`box_tracking_seconds` 和
`snapshot_build_seconds` 分别对应采集、机器人表面计算、深度修复、点云生成、
桌面分析、占据地图/OBB 构建和安全快照构建；地图更新、分块及编码耗时另见
`occupancy_update_seconds`、`occupancy_boxes_seconds`、`occupancy_encode_seconds`。
噪声排查另看 `raw_depth_filter_seconds` 与 `depth_evidence` 每路相机的
`isolated_pixels`、`robot_occluded_pixels`、`uncertain_raw_pixels` 和各类确认数量；
`diagnostics.occupancy.uncertain_raw_voxels` 表示本帧被拒绝深度留下的 unknown 体素。
`frame_interval_seconds` 表示实际
采集帧间隔；页面直接显示这些耗时。HTTP 响应的 `Server-Timing` 头另列
展示数据生成及 JSON 编码耗时。先比较这些耗时，再调整参数。
`--no-lingbot-depth` 可省去模型推理；若 `point_cloud_seconds` 或
`scene_analysis_seconds` 较大，可尝试 `--point-stride 4`，减少显示/桌面拟合及补全
点采样；地图开启时该参数不减少原始占据终点，关闭时也影响聚类输入。
若全图处理仍占主要时间，可在确认相机支持后尝试
`--width 640 --height 480`；降低分辨率后须重新验证小障碍物检测和标定效果。
提高 `--fps` 只改变相机采集上限，处理耗时不下降时不会提高实际发布帧率。
LeRobot 实机回放直接读取 RTDE/GET POS 反馈，在快照有效期内允许同一帧 OBB
用于多个控制周期的逐步复核；控制频率不再跟随检测帧率。
报告中的 `actions_executed`、`path_control_ticks` 和 `max_tracking_error_rad`
分别记录已完成数据集动作、路径控制周期和最大追踪误差；`phase=restore` 尚未进入数据集回放。
LeRobot 回放默认以 20 Hz 提供数据集动作，关节速度默认 0.5 rad/s、
加速度默认 1.2 rad/s²；碰撞复核、过期停止与 UR 自身保护继续生效。

相机启动时检查实际 SDK 流配置：默认保持 `1280×720@15` 彩色流；若某相机
不支持同分辨率的深度流，自动选用同帧率的深度配置，再由 SDK 对齐到彩色图。
实际序列号、USB 连接与 RGB/原始深度配置会打印，并写入快照 `camera_streams`。
当前 side D455（`260922303231`）在本次检查中为 USB 2.1，只提供
`1280×720@5` 的深度流，因此会选择 `640×480@15` 深度并对齐到 720p RGB。
如需原始 720p 深度 15 fps，需要恢复该相机的 USB 3 连接。
可显式指定 `--depth-width 640 --depth-height 480`，两项必须一起提供。
启动失败会显示具体相机和流配置，释放已启动的其他相机，并发布 error 状态。
若之后出现 `left_arm/right_arm: Active TCP differs from fixed-surface FK`，
则是控制器 active TCP 与 `robot.flange_to_active_tcp` 的一致性校验失败；
需要核对对应机械臂的实际工具/TCP 配置，不能据相机已启动判断空间对齐已通过。

QP 修改首动作的12个关节；修正后复核 FK 点轨迹及帧间线段，执行前复查新快照。
NaN、过期、不可控危险约束、复核碰撞拒绝执行；推理有超时，servo 有独立停止
看门狗。这是有限采样点的局部检查，不是未观测表面/动态环境的完整安全证明。

物理执行需 `--execute` 和精确确认串；`--execute-grippers` 可显式开启双
2F-85 输出（有约束时保持夹爪，发送前包含夹爪 FK 轨迹检查）。
新标定文件接入与离线几何校验已完成；受控实机验收尚未进行，本轮没有发送运动或夹爪命令。

## 验证与历史代码

无需训练动作/点流模型的 LeRobot 双臂动作回放验证见
[MODEL_FREE_OBSTACLE_VALIDATION.md](MODEL_FREE_OBSTACLE_VALIDATION.md)。
入口 `python -m obstacle_validation.lerobot_replay` 将 Quest3 HDF5 转成 LeRobot
数据集，并按绝对关节目标和双夹爪目标逐条回放；`avoid` 使用固定表面点 CBF
修正动作，复位阶段局部 CBF 无法前进时尝试经碰撞检查的 RRT 绕行。默认虚拟执行。
实机 `baseline` 可不启动障碍物服务，仅用双臂实时反馈回放数据集动作；
实机 `avoid` 需要 `--live` 的双视角检测服务。两者均需专用 `--execute-ack`。

```bash
conda run -n safety python -m pytest -q
```

逐项状态见 [VLAsafe_CODE_REVIEW.md](../VLAsafe_CODE_REVIEW.md)。
旧说明保留在 [docs/legacy](legacy/README_2026-09-22.md)。
LIBERO/Robosuite、旧 decoder、硬件维护与采集 SDK 仍有引用，见
[历史入口清单](legacy/INDEX.md)；没有批量删除这些文件。
