# VLSA、VLPSA 与 OA-VLA 独立测试

入口：`scripts/evaluate_libero_safety_methods.py`。支持 `--method vlsa`、
`--method vlpsa`、`--method oavla` 和 `--method all`；每个方法建立独立 socket
连接、创建独立仿真环境、重置控制器并输出逐 episode 结果。

当前实现是 **LIBERO-Safety 上共享感知和控制接口的适配版**。VLSA/VLPSA 原论文的
SafeLIBERO 是另一个基准。不能把此入口输出称为原论文复现成绩，也不能把随机模型或
Panda/Lift fixture 结果填作正式 LIBERO-Safety 成绩。

## 方法及协议

| 方法 | 本地安全过滤 | 策略来源 |
| --- | --- | --- |
| VLSA | 末端椭球、Eq. (9) barrier、每障碍两个虚拟分离面控制、硬 CBF QP | socket π0.5 或 OA-VLA 动作 |
| VLPSA | Poisson PDE、全身采样、抓持物采样注入、时变项、双分辨率 Boolean OR CBF、软 CLF 跟踪 | socket π0.5 或 OA-VLA 动作 |
| OA-VLA | 现有 `EllipsoidMCBF` 的全身约束、引导和有限运动检查 | socket Panda OA-VLA 动作 |

VLSA 控制器没有调用 OA-VLA 的规划器；VLPSA 也没有用椭球距离代替 Poisson 场。
Poisson 使用 CPU 稀疏共轭梯度，控制 QP 使用 SciPy SLSQP；OA-VLA 保留现有 OSQP。
求解失败、约束残差不合格或有限步检查失败时执行 hold，并记录原因。

共享实验协议：

- 策略输入为同一映射的四路 RGB、8 维 EEF 状态和语言；输出为 20 Hz、归一化
  OSC 语义的 7 维动作。仅执行预测动作块的第一条作为 50 ms 局部目标。
- 默认低层控制为 100 Hz；在两次策略推理之间跟踪同一位姿目标。三方法统一使用
  7 关节 `JOINT_VELOCITY`，保留 Panda 冗余自由度。夹爪为外生命令。
- VLSA 通过关节 Jacobian 修正末端位姿，保留末端约束范围；不是全臂 VLSA。
  多障碍逐个建约束属于适配扩展。若需要单障碍 VLSA 对照，只指定一个障碍。
- 感知使用**相同的仿真实例标签和同次渲染的深度**重建可见障碍边界，再统一加
  `--depth-padding`。不读取障碍真值位姿或网格作为控制输入。实例标签是额外输入，
  所以这是共享感知对照，不包含原始 VLSA 的 VLM/GroundingDINO 或 OA-VLA 的检测对比。
  原始 OA-VLA 预测 OBB/Voxblox 完整链路仍使用旧入口 `evaluate_oa_vla_libero_safety.py`。
- 全身采样使用机器人碰撞几何的保守包围盒表面网格，覆盖半径不超过
  `sample_spacing/sqrt(2)`；相同半径加 margin 用于膨胀 PSF 占据域。
  夹爪使用固定的开合扫掠包络，避免只保护当前开度。
- `grasp_objects` 明确列出可抓持对象。两侧指垫接触且命令闭合时，将可见 RGB-D
  包围盒加 2 cm 余量，转换至 EEF 坐标后注入样本；打开夹爪时清除。该接触传感是
  仿真代理。遮挡下的物体完整包围、刚性固定和速度跟踪都是理论保证的条件，当前
  实现不声称对不可见几何、滑移或动力学误差提供严格无碰撞保证。
- VLPSA 对粗、细 PSF **同时**施加论文 Boolean OR 的两组差值松弛约束，
  不是只选择最大场的梯度。细网格默认 2 cm / 0.5 m 窗口、粗网格 6 cm；
  这是可配置的 CPU 起点，不是论文 GPU 实时参数。真实耗时在逐步日志中记录。

论文依据：[VLSA IV-C](https://arxiv.org/html/2512.11891v2)、
[VLPSA II–III](https://arxiv.org/html/2609.22462v1)。

## 环境

仿真与模型服务分环境安装。仿真端使用 Python 3.11/3.12、NumPy<2、robosuite 1.4。
正式任务安装 LIBERO-Safety 仓库配套的 robosuite fork 和完整资产；准备步骤见
[本地部署](LIBERO_SAFETY_LOCAL.md#环境和场景资产)。仅安装 robosuite 包不足以运行正式任务。

```bash
python -m pip install -r requirements/libero_comparison.txt
export PYTHONPATH="$PWD/oa-vla/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/libero_comparison_numba
```

LeRobot 0.6 只安装在模型服务环境；无需在 robosuite 仿真环境安装 LeRobot。
OA-VLA 服务使用本项目固定的 Transformers 依赖。不要覆盖现有模型环境来满足另一端。

## 局域网模型服务

支持现有 `oa_vla_tcp_v1`：四路 RGB + state + prompt + 帧上下文，回复时原样回传
session/sequence/request_id/context。任意其他 socket 格式需要协议适配，不能直接连接。
元数据必须声明 Panda 8-state/7-action、`libero_osc_pose`、20 Hz 和相机映射；
未训练模型必须声明 `test_only=true`。

推理机器上启动 π0.5（适用于原生 LeRobot checkpoint）：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/serve_libero_safety_policy.py \
  --backend lerobot --model /models/pi05_libero_safety/pretrained_model \
  --normalized-osc-actions --device cuda --host 0.0.0.0 --port 8000
```

`--normalized-osc-actions` 确认训练数据已经转换动作单位；不是运行时缩放开关。
模型需包含保存的 processors 和本地可用 tokenizer。此服务不限制为旧入口的 LeRobot 0.4.4，
原生 LeRobot 0.6 的实际 checkpoint 加载仍需要在对应服务环境验证。

启动 Panda OA-VLA：

```bash
python scripts/serve_libero_safety_policy.py \
  --backend oa-vla --model /models/oavla_panda \
  --device cuda --host 0.0.0.0 --port 8001
```

未训练模型额外传 `--test-only`；OA-VLA 导出自带的 test_only 标记始终保留。
记录每种策略的 checkpoint、训练数据和预算；若 VLSA/VLPSA 与 OA-VLA 使用不同策略，
表格反映的是完整系统差异，不能仅归因于安全过滤器。

## 正式评测命令

修改 `configs/libero_safety_endpoints.example.json` 中的 IP，保存为自己的端点配置。
VLSA/VLPSA 可以共用 π0.5 服务；OA-VLA 可连接另一端口或另一台机器。
`configs/libero_safety_comparison.example.json` 提供任务清单格式。
障碍名与可抓持对象名必须逐任务核对官方 BDDL；代码遇到不存在的名字直接报错。
只有列在 `grasp_objects` 中的对象才有抓持物保护；空列表表示该机制在此任务未启用。

```bash
python scripts/evaluate_libero_safety_methods.py \
  --libero-root /data/LIBERO-Safety \
  --manifest configs/libero_safety_comparison.example.json \
  --endpoints /data/libero_endpoints.json \
  --method all --max-steps 3000 --control-freq 100 \
  --output outputs/libero_comparison_run01
```

单独运行时，将 `--method all` 换成 `vlsa`、`vlpsa` 或 `oavla`，并使用不同输出目录。
合并三个独立运行：

```bash
python scripts/summarize_libero_safety_methods.py \
  --runs outputs/vlsa_run01 outputs/vlpsa_run01 outputs/oavla_run01 \
  --output outputs/libero_comparison_table01
```

合并器检查协议参数、任务与初始化索引，拒绝重复方法和不同初始物理状态。
`--method all` 也会共享相同初始化，逐方法创建环境。输出目录必须是新目录。

## 指标与输出

| 指标 | 定义 |
| --- | --- |
| SR | 达成官方任务成功条件的 episode 数 / 全部请求的 episode 数 |
| SSR | 达成目标，且从初始状态起无官方安全约束违反、无禁止接触的 episode 数 / 全部请求数 |
| CR | 任意物理子步出现机器人或已抓持物与指定禁碰障碍的非正距离 MuJoCo 接触的 episode 数 / 全部请求数 |
| Steps | **成功 episodes** 的控制 tick 均值，包含 warmup；同时另报所有 episodes 的均值 |

CR 的禁碰对象由任务清单显式确定；未列入的环境/目标接触、同臂自碰不在这个 CR
口径内，不能把它命名为所有接触碰撞率。力约束等计入 SSR，单独的力违规不会自动记为 CR。
不以障碍位移代替接触，也不把所有安全违规统一称为碰撞。

碰撞/违规后继续运行到成功或超时，允许记录“成功但不安全”；warmup 中的违规也计入。
官方 horizon/done 不直接作为成功，成功使用 `_check_success()`。
默认 100 Hz 时，Steps / 100 才是仿真秒数；不同控制频率的 Steps 不可直接比较。
无成功 episode 时，Steps(success) 为 N/A，不填 0；Steps(all) 仍有值。
墙钟推理期间仿真暂停，不能把仿真时间当成真实端到端实时性能。

输出：

- `run_config.json`、`policy_<method>.json`：完整协议、参数与 socket 模型元数据。
- `<method>/<suite>_L<level>_T<task>/episode_*/steps.jsonl`：名义动作、实际关节动作、
  QP/Poisson 诊断、hold 原因、碰撞、时延。
- 同目录的 `report.json`：成功、碰撞、违规、步数、初始化 hash；异常另有 `error.txt`。
- `summary_<method>.json`：episode 明细、SR/SSR/CR、Wilson 95% 区间、
  成功 Steps 均值与 t 区间、全样本 Steps。
- `comparison.json`、`table.csv`、`table.md`：总体和分套件表格，可用于论文制表。

只要有错误或缺少请求的 episode，主要指标为 null/空白，不能把漏跑样本悄悄移出分母。
`test_only=true` 或 fixture 时，`paper_eligible=false`。该字段只检查技术前提，
不是对实验划分、训练公平性或论文结论的自动认证。

## 无正式场景和 checkpoint 时的实测

```bash
python scripts/smoke_libero_safety_methods.py \
  --output outputs/libero_methods_smoke_run01 --episodes 2 --steps 20
```

脚本实际创建并保存未训练的 tiny Panda OA-VLA checkpoint，开启 127.0.0.1 随机端口的
真实 TCP 服务；三方法使用同一模型、确定性输入相关采样噪声和相同 Panda/Lift 物理初始状态。
小幅随机动作的 std=.03 写入测试 checkpoint，避免所有请求仅因动作越界被拒绝。
没有训练，不提供任务能力证据。结果位于 `<output>/results/table.csv`，标为 test_only。

fixture 使用 20 Hz、2 warmup ticks 和用户指定控制 ticks。20 控制 ticks 超时对应
Steps(all)=22，Steps(success)=N/A。不能把该结果当作 LIBERO-Safety 或原论文成绩。

针对性验证：

```bash
RUN_LIBERO_COMPARISON_SIM=1 python -m pytest -q \
  verification/test_libero_comparison.py verification/test_libero_comparison_sim.py \
  verification/test_vlsa_mcbf.py verification/test_libero_safety.py
```

包含 Poisson 离散 PDE/梯度/膨胀、Boolean OR 两级约束、时变项、硬 CBF 与软 CLF、
VLSA 安全修正、双指接触触发、抓持物变换与释放、真实 RGB-D 投影与 FK 状态恢复、
碰撞累计和统计分母检查。
