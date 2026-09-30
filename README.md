# Surface Action Model

基于 OpenPI PI0.5 的双臂 action / surface 联合预测。方法类为
`SurfaceActionModelPytorch`；保留上游 `PI0Pytorch`、`pi05_base` 名称。

```text
Quest3 HDF5 → FK 可见性筛选 → CoTracker + 实测深度 → surface/action shard
RGB + task + state → 冻结 PI0.5 VLM → action expert（训练、flow matching）
VLM 输出 prefix + current_surface + action_chunk → point DiT → point decoder
    → flow velocity → Euler 积分 → surface_chunk
front/side D455 → LingBot 修复 → 双臂剔除 → 聚类 OBB
    → 未来 surface 检查 → 首动作关节 CBF-QP → FK 轨迹复核
```

点分支以同一次生成的 action chunk 为额外条件，使随机点预测与动作对应。
训练只冻结 VLM；action expert、动作投影和 point DiT/decoder 都参与训练。

## 当前硬件与标定

左右工具均为 Robotiq 2F-85。只读枚举确认：

双臂目前已断开；下列相机枚举来自此前检查，不表示整套硬件当前可运行。

| 角色 | 型号 | 序列号 |
| --- | --- | --- |
| front | D455 | 260922305390 |
| side | D455 | 260922303231 |
| left_wrist | D435i | 420222071502 |
| right_wrist | D435i | 405622072940 |

**front 旧外参已失效，side 尚未标定。** 当前
`safety.camera_calibration_valid=false`，在线感知/控制入口拒绝启动。
不能只修改标志跳过标定。流程见 [CALIBRATION.md](docs/CALIBRATION.md)。
保留的历史外参仅适用于匹配旧采集时相机位置的数据。

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
  --calibration configs/hardware/calibration/left_base_to_front_camera.yaml \
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

默认 `.shard` 是可 mmap 的 NPY 数组目录，也支持旧 NPZ。旁边保存原始 HDF5
完整 `.source.hdf5` 副本及相对引用。每帧提供 H 步窗口；末尾 padding 有
`action_step_mask` 和点 mask，训练排除无效监督。
`--complete-windows-only` 保留旧窗口行为。

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

默认 dry-run。快照为 `http://127.0.0.1:8766/snapshot.json`，包含双视角 OBB、
双臂 qpos、双夹爪实测状态与 current_surface。使用实际流内参；LingBot 修复
后重新剔除双臂，并保留原始深度支持域限制。

QP 修改首动作的12个关节；修正后复核 FK 点轨迹及帧间线段，执行前复查新快照。
NaN、过期、不可控危险约束、复核碰撞拒绝执行；推理有超时，servo 有独立停止
看门狗。这是有限采样点的局部检查，不是未观测表面/动态环境的完整安全证明。

物理执行需 `--execute` 和精确确认串；`--execute-grippers` 可显式开启双
2F-85 输出（有约束时保持夹爪，发送前包含夹爪 FK 轨迹检查）。
新标定与受控实机验收尚未完成，本轮没有发送运动或夹爪命令。

## 验证与历史代码

```bash
conda run -n safety python -m pytest -q
```

逐项状态见 [VLAsafe_CODE_REVIEW.md](VLAsafe_CODE_REVIEW.md)。
旧说明保留在 [docs/legacy](docs/legacy/README_2026-09-22.md)。
LIBERO/Robosuite、旧 decoder、硬件维护与采集 SDK 仍有引用，见
[历史入口清单](docs/legacy/INDEX.md)；没有批量删除这些文件。
