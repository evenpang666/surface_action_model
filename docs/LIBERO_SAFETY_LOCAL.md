# LIBERO-Safety 本地推理

本入口复用当前真机的 OA-VLA TCP 协议、`DynamicOBBFilter`、Voxblox、双框交集、透明障碍补全和
`EllipsoidMCBF`。每次只执行动作块的第一条，推进一个 20 Hz 仿真 tick，再获取新图像。
入口仅创建仿真环境，不加载 UR 控制或 RealSense 设备。

## 先选择权重类型

| 权重 | 入口 | 能力 |
| --- | --- | --- |
| 本项目训练的 Panda OA-VLA format-2 | `--backend oa-vla`，连接 `oa-vla-serve` | 动作 baseline、完整感知与 MCBF 避障 |
| 原生 LeRobot π₀.₅ 的 `pretrained_model/` | `--backend lerobot --mode baseline --policy ...` | 两视角动作 baseline；可采集四视角训练数据 |
| 当前双 UR7e OA-VLA 权重 | 不兼容 Panda 输入输出 | 必须训练 Panda profile，不能截断 14 维动作或伪造元数据 |
| 官方 `LIBERO-Safety/pi05_libero_safety` | 官方 OpenPI/JAX 格式 | 不是上述两种导出；此入口不直接加载 |

Panda OA-VLA 的状态为 `[eef_xyz(3), eef_axisangle(3), finger_qpos(2)]`，8 维；
旋转使用 robosuite `quat2axisangle`，与数据导出器一致，不能随意改为 Euler 或强制主值旋转向量。
动作为归一化 OSC_POSE 的 `[dx,dy,dz,drx,dry,drz,gripper_command]`，7 维。
夹爪最后一维是有符号开合命令，不是位置目标，也不是关节角。

默认四个模型槽位对应四台不同的仿真相机：

```json
{"front":"agentview","side":"sideview","right_wrist":"robot0_eye_in_hand","left_wrist":"frontview"}
```

`left_wrist` 在单臂实验中保留模型槽位名称，实际接第二台外部相机 `frontview`；没有第二只手臂。
这四个视角都真实渲染，不复制图像。RGB 统一为左上原点；RGB 与深度同时纵向翻转一次。
只有 front/side 进入检测与深度建图。四视角 OA-VLA 权重必须按同一映射训练，handshake 会校验映射。
需要其他映射时，用 JSON 文件传 `--camera-map`，同时重新准备一致的训练数据。

## 环境和场景资产

所有命令在项目根目录执行。使用单独环境，避免覆盖原有 LIBERO 和真机环境。
以下代码版本已核查：`19ec8df23eedfbb9265bafd3e56495fcebfcfcd0`。

```bash
conda create -n libero-safety-sim python=3.11 -y
conda activate libero-safety-sim
python -m pip install -r requirements/libero_safety.txt
python -m pip install -e .
python -m pip install --no-deps -e './oa-vla[client]'

mkdir -p outputs/deps
git clone https://github.com/LIBERO-SAFETY/LIBERO-Safety.git outputs/deps/LIBERO-Safety
git -C outputs/deps/LIBERO-Safety checkout 19ec8df23eedfbb9265bafd3e56495fcebfcfcd0
python -m pip install -e outputs/deps/LIBERO-Safety
python -m pip install -e outputs/deps/LIBERO-Safety/third_party/robosuite-1.4
python -m pip install 'huggingface_hub[cli]>=0.34,<1'

hf download LIBERO-Safety/libero_safety_assets --repo-type dataset \
  --local-dir outputs/downloads/libero_safety_assets
unzip outputs/downloads/libero_safety_assets/assets.zip \
  -d outputs/deps/LIBERO-Safety/libero/libero/
```

解压后应为 `outputs/deps/LIBERO-Safety/libero/libero/assets/`，内有场景 XML 和模型资源。
资产压缩包约 10.7 GB，预留足够解压空间。部分官方对象类直接按 checkout 的相对路径找资产，
因此资产必须放在该位置；外置磁盘可将这个 `assets/` 链接到外部目录。
不要只改全局 `~/.libero/config.yaml`。

Linux 的 wand/ImageMagick、渲染及 Voxblox 构建还需要系统库；例如：

```bash
sudo apt-get install libmagickwand-dev libegl1 libgl1 libexpat1 \
  libfontconfig1-dev cmake build-essential libeigen3-dev libgoogle-glog-dev \
  libprotobuf-dev protobuf-compiler
python scripts/build_voxblox.py
export MUJOCO_GL=egl
```

无 EGL 的机器可尝试 `MUJOCO_GL=osmesa` 并安装 OSMesa。Voxblox 缺失会明确报错，不会静默切换算法。
本地入口创建独立 `outputs/libero_safety_config/config.yaml`；`LIBERO_CONFIG_PATH` 在官方代码中是**目录**。

```bash
python scripts/evaluate_oa_vla_libero_safety.py \
  --libero-root outputs/deps/LIBERO-Safety --check-only
```

`--check-only` 检查路径、依赖及所选任务文件，不加载模型或创建仿真；也不能代替实际渲染验证。

## 当前自定义模型的完整链路

在 OA-VLA 模型环境启动服务，和真机使用同一个服务入口：

```bash
oa-vla-serve --model /models/oavla_panda --device cuda --port 8000
```

在仿真环境：

```bash
python scripts/evaluate_oa_vla_libero_safety.py \
  --libero-root outputs/deps/LIBERO-Safety \
  --suite obstacle_avoidance --level 0 --task 0 --episodes 0 1 2 \
  --host 127.0.0.1 --port 8000 --mode safety \
  --output outputs/libero_safety/panda_safety_run1
```

同一 checkpoint 关闭安全层的对照：

```bash
python scripts/evaluate_oa_vla_libero_safety.py \
  --libero-root outputs/deps/LIBERO-Safety \
  --suite obstacle_avoidance --level 0 --task 0 --episodes 0 1 2 \
  --mode baseline --output outputs/libero_safety/panda_baseline_run1
```

支持 `obstacle_avoidance`、`obstacle_avoidance_human`、`affordance`、`human_safety`，
等级 0/1/2，等级内任务 ID 0..4。语义安全拒答轨道没有接入动作 rollout。
几何安全层不能实现安全抓取区域推理或力约束控制，因此优先在两个避障套件分析效果。
`--episodes` 是官方初始化列表里的索引，不是训练数据的 episode 编号。

当前 Panda 模型导出应包含训练自动产生的：

```json
{
  "action_semantics":"libero_osc_pose",
  "state_semantics":"libero_eef_axis_angle_gripper",
  "image_convention":"opencv_top_left",
  "camera_names":{"front":"agentview","side":"sideview","right_wrist":"robot0_eye_in_hand","left_wrist":"frontview"},
  "action_period_s":0.05,
  "detection_trained":true
}
```

这些字段必须来自相同 profile 的训练，不应手工改动双 UR7e 权重来绕过校验。
随机模型只能显式用于仿真 baseline 的通信测试：

```bash
oa-vla-init-test --profile libero-safety --output outputs/panda_socket_test/policy
oa-vla-serve --model outputs/panda_socket_test/policy --device cpu --num-steps 1
# 在仿真终端运行，输出不代表任务能力：
python scripts/evaluate_oa_vla_libero_safety.py \
  --libero-root outputs/deps/LIBERO-Safety --mode baseline --allow-test-policy \
  --max-steps 3 --no-video --output outputs/panda_socket_test/rollout
```

## LeRobot baseline 与四视角训练采集

原生 LeRobot 的训练、数据转换及环境配置见 [集群训练](LIBERO_SAFETY_CLUSTER_TRAINING.md)。
用该文档训练得到的 checkpoint，在安装了固定 LeRobot `[pi]` 依赖的独立仿真环境运行：

```bash
python scripts/evaluate_oa_vla_libero_safety.py \
  --libero-root outputs/deps/LIBERO-Safety \
  --backend lerobot --mode baseline \
  --policy /models/pi05_safety/pretrained_model --device cuda \
  --suite obstacle_avoidance --level 0 --task 0 --episodes 0 \
  --output outputs/libero_safety/lerobot_baseline_run1
```

本入口使用保存的 pre/post processors 进行归一化、tokenizer 和反归一化；每次丢弃动作队列，重新观测。
只接受两路标准键 `observation.images.image/image2`、8 维状态、7 维动作的 native pi05。
必须先按集群文档把官方数据的**米制动作转换为归一化 OSC 动作**；不能直接用原始米制动作训练后套此入口。

官方数据缺少 side/frontview，不能用复制图像的方法训练当前四视角检测模型。
可先训练两视角 baseline，再用它采集真实渲染的四视角、安全完成的 rollout：

```bash
python scripts/evaluate_oa_vla_libero_safety.py \
  --libero-root outputs/deps/LIBERO-Safety \
  --backend lerobot --mode baseline --policy /models/pi05_safety/pretrained_model \
  --suite obstacle_avoidance --level 0 --task 0 --episodes 0 1 2 3 4 \
  --record-training outputs/datasets/panda_oavla_train \
  --label-obstacles bottle_of_perfume_1 white_place_box_1 \
  --training-horizon 8 --output outputs/libero_safety/collect_run1
```

上面障碍名称仅适用于所固定版本的 `obstacle_avoidance/L0/task0`；其他任务需按 BDDL 的
`:constraints` 明确指定障碍实例，不能把抓取目标全部标为障碍。
分割真值仅生成训练用的可见 OBB、跨视角实例 ID 和不可见标记；正常推理使用模型预测，不读取障碍真值。
失败/碰撞/超时 episode 不导出动作示范；成功数量可能远少于请求数量。L2 禁止通过该入口导出训练数据。
请另留 L0/L1 初始化索引作验证/测试，不要在训练采集索引上报告泛化成绩。

每个成功 episode 输出 `.shard/`，可直接交给 `oa-vla-train`；记录的动作是实际交给 `env.step` 的命令，
不是测得的关节增量。长 episode 四路 RGB 会占用较多内存，建议从 256 分辨率和逐 episode 调用开始。

## 日志、指标与可比性

输出包括 `run_config.json`、`summary.json`，以及每 episode 的 `steps.jsonl`、`report.json`、视频。
记录原动作、修正动作、hold、配对几何、QP 诊断、耗时和官方约束违规。

- `safe_success_rate`：安全且完成任务的比例；这是论文物理轨道 SR 对应的本地口径。
- `safety_violation_rate`：任意官方约束违反的比例；可能包含碰撞、力等，不应统一称 CR。
- episode 的 `success` 单独表示逻辑目标完成；成功同一步出现违规仍是安全失败。
- 检查每个物理子步的每个官方 predicate，不以对象位移替代碰撞；违规后终止 episode。
- warmup 也监测违规；异常运行保留错误日志，不能当作正常失败或从分母悄悄删除。

真机和仿真复用相同的框筛选、补全与 MCBF 求解代码。Panda 适配使用当前 Jacobian 将归一化 OSC 命令
映射到预测关节变化，并从 Panda 碰撞几何获得机器人椭球。它是局部运动学近似，实际执行仍由官方 OSC
动力学控制器完成，不能继承真机关节执行的精确对应关系或宣称严格无碰撞保证。
夹爪是外生开合命令，QP 不优化它；没有加入同臂自碰撞或连续轨迹扫掠证明。

仿真在推理等待期间暂停；动态物体按仿真时钟运动，障碍速度估计也使用该时钟。wall latency 另行记录。
保留与真机相同的 `--max-result-age 0.75` 默认值；慢速调试可调大并记录，不能把暂停仿真误当作实时性能。
空框、单侧不可见、过期结果或不合格的 barrier/slack 会 hold；hold 推进零位姿/零夹爪命令的一个 tick。
所有点的数值坐标均为 MuJoCo world；共享旧 snapshot 内的 `left_base` 标签是内部兼容字段，报告明确标为 world。

四视角、深度、安全层和更密集的子步检查属于本地扩展协议。论文结果比较时必须写明这些额外输入及协议差异，
两个模式使用完全相同的任务、初始化、时限和约束判定。当前没有声称完成官方 75 任务全量复现。

## 已完成的验证范围

有针对性的测试覆盖 profile 拒绝、动作第一步、瞬时接触累计、成功步违规、通用 MCBF 维度、训练 shard 读取、
字段转换。另在实际 Panda/Lift 场景验证四路 RGB-D、相机标定、自身掩码、FK 状态恢复和原生 Voxblox 筛选链路。
这些是适配层测试，不是 LIBERO-Safety 成绩；正式任务仍需下载完整资产和准备经过训练的相应权重。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=oa-vla/src python -m pytest -q \
  verification/test_libero_safety.py verification/test_oa_vla_runtime.py \
  verification/test_ellipsoid_constraint_scope.py verification/test_ellipsoid_replay.py
# 实际渲染测试；需在已安装官方 robosuite fork 和 Voxblox 的仿真环境运行：
MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=oa-vla/src python -m pytest -q \
  verification/test_libero_safety_sim.py
```

官方资源：[代码](https://github.com/LIBERO-SAFETY/LIBERO-Safety)、
[资产](https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets)、
[论文](https://arxiv.org/abs/2606.23686)。
