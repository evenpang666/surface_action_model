# Dual UR7e Safety World-Action Model

本项目的默认部署场景是当前 Quest3 双臂工作站：左 UR7e + Robotiq 2F-85、右
UR7e + Robotiq EPick、单台 front D455，全部安全几何统一到 `left_base`。

唯一主流程为：

```text
Quest3 HDF5
  → 固定身份真实碰撞网格点流
  → PI05 联合预测 action chunk + 未来机器人表面点流
  → front D455 + LingBot-Depth 实时桌面障碍物 OBB
  → 12-DOF CBF-QP 修正双臂关节 action
```

项目默认配置在 [configs/ur7e_dual_quest3.yaml](configs/ur7e_dual_quest3.yaml)。
环境统一使用 `conda safety`；OpenPI 训练/服务也可以从 `openpi` 项目用 `uv run`。

## 0. 当前安全前置条件

左右臂 `flange→active_tcp` 均已于 2026-09-10 通过只读 RTDE 连续 30 次实测，
结果保存在 `outputs/calibration/dual_flange_to_active_tcp.json` 并写入项目配置。
若 pendant active TCP、转接板或末端工具发生变化，请重新运行（不会创建
RTDE control）：

```bash
conda run -n safety python real_scripts/read_dual_active_tcp_calibration.py
```

未核验 TCP 时，双臂安全标签生成和实机执行会自动被阻止。

## 1. Quest3 HDF5 预处理

单臂 episode 默认生成 7D 动作（6 关节 delta + 1 夹爪目标），双臂 episode
自动生成 14D 动作。若要以左臂遥操数据训练部署用的双臂动作接口，显式传
`--pad-inactive-right-arm`：右臂状态/动作补零并表示不下发命令，点流仍只使用
真实左臂表面，绝不伪造未知右臂位姿的网格点。

```bash
conda run -n safety python scripts/preprocess_quest3_hdf5.py \
  --episode ../quest3_collect/data/left_vr_episodes/episode_00008.hdf5 \
  --output outputs/quest3_pi05/episode_00008.npz \
  --pad-inactive-right-arm
```

输出包含：

- `rgb_*`、`qpos`、`gripper_position`；
- `action_chunks[N,H,7|14]` 与明确的 `action_layout`；
- `fixed_link_points[T,L,P,3]`、`point_ids`；
- `current_link_points[N,K,3]`；
- `target_point_offsets[N,H,K,3]` 和监督 mask；
- 坐标系、网格哈希、左右 base 变换、动作来源等审计字段。

如需把已有 CoTracker 实测点作为可选辅助监督，可加：

```bash
--measured-flow-npz outputs/episode_00008_left_surface/episode_00008_left_ur7e_2f85_cotracker1024_measured_surface.npz
```

默认安全训练目标仍是固定身份碰撞网格点。CoTracker 点会缺失且与 FK
Jacobian 没有一一对应关系，不应直接作为在线 CBF 控制点。

## 2. 联合训练 PI05

项目使用 OpenPI JAX `pi05_base` 作为唯一预训练来源。首次使用时安装仓库随附的
Transformers AdaRMSNorm/KV-cache 补丁，并执行低内存 bfloat16 转换：

```bash
conda run -n safety bash -lc 'cp -r \
  openpi/src/openpi/models_pytorch/transformers_replace/. \
  "$CONDA_PREFIX/lib/python3.11/site-packages/transformers/"'
PYTHONPATH=openpi/src:openpi/packages/openpi-client/src \
conda run -n safety python openpi/examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --config-name pi05_aloha \
  --output-path outputs/pretrained/pi05_base_pytorch \
  --precision bfloat16
```

转换器会校验所有基础权重，并复制原检查点的 normalization assets。不要将
LeRobot 缓存或 JAX `params/` 目录直接传给 PyTorch 训练器。

每个新 Quest3 HDF5 的根属性 `task_description` 会自动写入 shard 的
`task_text`，并在训练时逐 episode tokenization；`--task` 仅用于有意覆盖，
旧 HDF5 才回退到项目配置中的通用任务文本。

先只检查 shard 契约，不加载大模型：

```bash
conda run -n safety python scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/quest3_pi05/episode_00008.npz \
  --output outputs/pi05_quest3/left_joint_safety.pt \
  --validate-only
```

正式训练：

```bash
uv run --project openpi scripts/train_pi05_ur7e_surface_pytorch.py \
  --dataset outputs/quest3_pi05 \
  --output outputs/pi05_quest3/dual_joint_safety.pt \
  --max-points 256 \
  --epochs 20 --max-steps 20000
```

### 4 × V100（16 GB）分布式训练

V100 不支持硬件 BF16；当前训练器的 DDP 模式会复制而非分片 PI0.5 主干，
因此 16 GB 卡只能训练新增的安全点流层。以下命令的全局 batch 为
`4 × 4 × 2 = 32`，并默认开启 activation checkpointing：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_pi05_ur7e_surface_pytorch.py \
  --distributed \
  --dataset outputs/quest3_pi05/left_dual_formal \
  --output outputs/pi05_quest3/left_dual_pi05_v100_ddp.pt \
  --epochs 200 --max-steps 2000 --batch-size 4 --gradient-accumulation-steps 2 \
  --max-points 256 --precision float16 --freeze-base
```

完整 PI0.5 AdamW 微调仍需要 FSDP/ZeRO-3 参数、梯度和优化器状态分片；4 张
16 GB V100 即使使用 DDP 也不会获得模型显存分片。

### A100 / H100 多卡训练

脚本的 DDP/NCCL 路径同样适用于 A100（SM 8.x）和 H100（SM 9.x）。对于
80 GB 型号，可以进行完整 PI0.5 微调；下例为 4 卡、全局 batch 16（每卡 4）的
单机启动方式：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_pi05_ur7e_surface_pytorch.py \
  --distributed \
  --dataset outputs/quest3_pi05/left_dual_formal \
  --output outputs/pi05_quest3/left_dual_pi05_a100_h100_full.pt \
  --epochs 200 --max-steps 2000 --batch-size 4 --gradient-accumulation-steps 1 \
  --max-points 256 --precision bfloat16
```

40 GB A100 不应使用上面的全量 AdamW 模式；使用 V100 段落中的
`--freeze-base` 配置即可，但可将 `--precision` 改为 `bfloat16`。多节点时，将
`--standalone` 换为集群分配的 `--nnodes`、`--node_rank`、`--master_addr` 与
`--master_port` 参数；每个节点都必须可见相同的 checkpoint、tokenizer 与数据路径。

同一 checkpoint 中的 shard 必须具有相同的臂数、action 维度、horizon 和点
布局；不要混合 7D 单臂与 14D 双臂数据。左臂数据使用占位模式后应全部采用
14D shard。默认只使用 front D455，与实际推理
一致；若要增加 wrist 视角，请在训练时显式传 `--camera-map`，并确保部署端也
提供相同视角。

checkpoint 保存 action/qpos 归一化统计、逻辑 action 维度、点子集索引、相机
映射和坐标系。PI05 内部保留 32 个 action 槽，但只向执行器返回 7 或 14 个
有定义的量。

## 3. 联合推理服务

```bash
uv run --project openpi scripts/serve_quest3_pi05_safety.py \
  --checkpoint outputs/pi05_quest3/dual_joint_safety.pt \
  --device cuda \
  --port 8000
```

每个请求一次性返回：

- `actions[H,14]`；
- `point_offsets[H,K,3]`；
- `predicted_robot_points[H,K,3]`；
- 与训练一致的 `selected_point_indices`。

旧的 prefix-token + 独立 safety decoder 服务不属于当前部署主线。

## 4. 实时桌面障碍物

该服务是 front D455 和双臂 RTDE receive 的唯一所有者，只读机器人，不发送
运动或夹爪指令：

```bash
conda run -n safety python real_scripts/live_dual_ur7e_obstacle_model.py
```

浏览器界面为 <http://127.0.0.1:8766>，安全快照为
`http://127.0.0.1:8766/snapshot.json`。v2 快照包含 front RGB JPEG、双臂 qpos、
采集单调时钟、`left_base` 坐标系以及每个 OBB 的 center/axes/half-sizes。

障碍物链路为：机器人真实网格深度剔除 → LingBot-Depth 修复 → 原始 D455
支持域限制 → 桌面 RANSAC → 离群点剔除 → 精确点级 DBSCAN → 桌面连接检查
→ 时序 OBB 确认。界面可用按钮或空格暂停显示；暂停不停止后台安全快照更新。

## 5. 双臂 CBF-QP

先运行 dry-run；它不会建立 RTDE control 连接：

```bash
conda run -n safety python real_scripts/run_ur7e_vla_safety_executor.py \
  --prompt "manipulate the tabletop instruments safely" \
  --once
```

执行器对预测点流建立两类约束：

1. 未来固定表面点与桌面 OBB 的 signed-distance barrier；
2. 左右臂预测表面点对的最小间距 barrier。

有限差分 Jacobian 与两臂 12 个关节同时进入 bounded least-change QP。快照过期、
坐标系错误、服务契约错误或 QP 不可满足时均 fail closed 为双臂 hold。

物理执行需要右 TCP 已核验，并同时提供 `--execute` 与命令行显示的精确确认串。
当前执行器只发送两臂 `servoJ`，不会猜测 EPick 的真空 I/O，也不会发送任何
夹爪命令；配置并验证 EPick/2F-85 控制接口后才能补上该部分。

## 6. 一键契约检查

```bash
conda run -n safety python scripts/validate_project_pipeline.py \
  --episode ../quest3_collect/data/left_vr_episodes/episode_00008.hdf5 \
  --shard outputs/quest3_pi05/episode_00008.npz
```

还可以传 `--checkpoint` 和 `--obstacle-url`，逐层检查训练 checkpoint 与实时
快照。返回码 0 表示所检查层级通过；未核验 TCP 默认产生返回码 2。

## 目录

```text
configs/                  当前双臂项目配置
safety_module/            配置与学习模块
scripts/                  HDF5 预处理、训练、服务、契约验证
real_scripts/             双臂网格、实时障碍物、CBF 与执行器
assets/robot_models/      UR7e、2F-85、EPick 网格/URDF
openpi/                   上游 OpenPI 与本项目 PI05SafetyPytorch 扩展
outputs/                  本地预处理、可视化与 checkpoint（不自动删除）
```
