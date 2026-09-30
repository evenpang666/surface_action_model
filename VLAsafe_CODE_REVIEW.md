# VLAsafe 项目代码审查记录

审查日期：2026-09-22。目标规范以本次用户描述为准，现有 README 不作为正确性标准。

原审查轮仅新增记录；后续已按用户要求修改实现。下文第 1–6 节保留原始问题和用户修改的命名要求，历史行号可能已变化；当前进度以本节修复状态为准。审查包含工作区尚未提交的修改：`dual_ur7e_surface.py`、`live_dual_ur7e_obstacle_model.py`、`run_ur7e_vla_safety_executor.py`、两个主线训练/服务脚本，以及未跟踪的 legacy 入口。

## 修复状态（2026-09-22）

用户确认左右均为 Robotiq 2F-85，front 旧标定失效、side 未标定；随后告知双臂已断开。因此停止硬件连接检查，未下发运动或夹爪指令。当前配置明确禁止未标定的在线感知/执行。

| 原记录 | 本轮软件改动 | 验收状态/剩余工作 |
| --- | --- | --- |
| P0-01 | 恢复 meta 加载后的 Gemma RoPE / SigLIP 非持久化 buffer，未知 buffer 报错 | 小型真实模型普通加载与低内存加载输出一致；完整权重待验 |
| P0-02 | 推理超时、推理后及执行前时效/状态漂移复查、独立 servo 停止看门狗 | 离线测试；通信中断与停止时延待真机验收 |
| P0-03 | 快照、预测、索引、OBB、QP 的有限性/形状/时间契约检查 | 离线异常测试通过 |
| P0-04 | 每点每盒保留最深危险帧，拒绝不可控危险约束 | 离线回归通过 |
| P0-05 | 双 2F-85 配置/网格/实测夹爪状态统一，显式可选双夹爪输出 | 当前 TCP 无法重新核验；双臂恢复后再检查 |
| P1-01 | 双臂 CoTracker 合并、可见性 mask、在线实测 current_surface、实测点与 link/Jacobian 关联 | 合成数据测试；完整双臂实测 episode 尚未跑通验收。录制 extractor 目前只用 front，两臂需有初始可见种子 |
| P1-02 | 新 SurfaceActionModelPytorch：prefix/current_surface/action_chunk 条件 point DiT + decoder，速度目标与多步 Euler 积分 | 小型真实 backbone 前后向/积分测试；需要重新训练，旧回归头权重不可直接迁移 |
| P1-03 | 默认只冻结 VLM，action expert 与 point module 保持训练 | 梯度测试；大模型显存及完整训练待验 |
| P1-04 | front+side 采集/融合、双视角 LingBot、修复后重新剔除双臂、实际流内参及时间偏差检查 | 模拟双视角链路通过；真实双视角标定和 LingBot 联调待完成 |
| P1-05 | 每帧 H 窗口+padding/mask，保存原 HDF5 副本、commanded/executed 与动作周期 | 合成 HDF5→shard→训练读取测试；不规则周期明确拒绝，未实现自动重采样 |
| P1-06 | 点数按实际 surface_arm_count，保留单臂补位数据诊断兼容 | 真实执行要求双臂实测新模型 |
| P1-07 | 当前 SDF 法向、首动作 QP 后全 chunk FK 子步/线段复核、发送前最新状态复核 | **部分缓解，不是完整安全证明**；有限采样、模型误差及动态环境仍需实机评估 |
| P2/路径环境 | mmap shard、共享 tokenizer、FP16 master 参数与 GradScaler、配置本地化、进程内 HF 扩展、分环境依赖清单 | 新环境尚未全新安装验证，依赖清单不是完整传递锁；RGB/点懒加载，但状态统计仍占内存 |
| 名称 | 新方法类、current_surface/surface_chunk、live_obstacle_detection 与 viewer 入口 | 保留上游名称及兼容旧入口，不对历史权重强制改名 |
| 冗余 | 删除部分无效几何参数/重复过滤/重复 tokenizer；旧说明归档并建立历史入口索引 | 有引用的旧模型、采集和第三方代码暂未删除，避免误删用户工作 |

验证入口：`conda run -n safety python -m pytest -q`，测试位于 `verification/`。覆盖小型真实 Transformer、meta 恢复、点分支梯度/积分、数据 mask、QP 异常、看门狗、标定安装器和模拟双视角。使用现有 safety 环境；不代表新依赖环境、完整 CoTracker/LingBot/PI05 GPU 链路或物理控制已验收。当前操作说明见根 README 和 `docs/CALIBRATION.md`。

本轮最终离线回归结果：**21 passed in 4.96s**；`git diff --check` 通过。未运行完整模型训练或真机闭环。

后续硬件恢复顺序：确认标定板参数 → front/side 与双基座重新标定 → 双臂 TCP/夹爪只读核验 → 感知和策略 dry-run → 人工监督的低速受控实机验收。历史标定只用于确认匹配当时相机位姿的旧数据，不可用于当前在线控制。

范围：全仓结构、入口和依赖检索，重点逐段检查 Quest3 预处理、两种点模型、PI05 扩展、训练/服务、在线障碍物与 CBF 执行链路。对 168 个项目/相关 OpenPI Python 文件做 AST 语法检查。没有逐行审计 libsurvive、LingBot、LIBERO、厂商 SDK 等全部第三方实现；没有验证完整训练、GPU 大模型推理、实机延迟与物理效果。因此本记录不是“全仓所有行为已验证”的证明。

## 1. 总体结论

当前主线实现的是 **FK 网格固定点监督 + PI05 action expert 隐状态上的位移回归头 + 单 front 障碍物服务 + 首动作关节 QP**，与目标 **双臂 CoTracker 实测点 + prefix 条件 point DiT/decoder flow matching + front/side 重建** 存在结构性偏差。

| 环节 | 当前实现 | 判定 |
| --- | --- | --- |
| Quest3 HDF5 输入 | 支持 6/12 关节，输出 7/14 维动作 | 基本具备 |
| 双臂 CoTracker 实测表面标签 | 主线由 FK 网格生成；CoTracker 合并限制单臂 | 不符合 |
| H 步动作/表面序列 | `action[t:t+H]` 对应 `surface[t+1:t+H+1]` | 索引合理，动作语义和时间间隔需明确 |
| PI05 backbone | `Pi0Config(pi05=True)` | 符合 |
| prefix → point DiT → decoder | 主线为 action suffix 特征加点/关节 MLP | 不符合 |
| 点 flow matching / 迭代积分 | 主线为位移 MSE，一次输出点位移 | 不符合 |
| 仅冻结 VLM | `--freeze-base` 时符合，默认不冻结 | 默认不符合 |
| front + side 障碍物重建 | 主入口只有 front | 不符合 |
| LingBot、机器人剔除、聚类 OBB | 已实现，但掩码顺序和双视角不同 | 部分符合 |
| 策略端 192.168.124.64 | 可通过参数连接，默认 localhost | 部署配置未落实 |
| 预测未来点触发首动作 QP | 已实现，仅优化 12 个关节 | 部分符合，另有正确性问题 |

## 2. 逻辑偏差与缺陷

优先级：P0 为可能破坏预测/控制有效性的缺陷；P1 为方法核心偏差或契约断裂；P2 为可维护性与次要完整性问题。优先级不代表已经发生实机事故。

### P0-01：服务加载没有恢复非持久化缓冲区

- 位置：`scripts/serve_quest3_pi05_safety.py:130–139`；`openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py:143–145`；同目录 SigLIP 实现的 `position_ids`。
- 先在 meta device 构造，再 `to_empty`，最后加载 state dict。`inv_freq`、`position_ids` 为 `persistent=False`，不在 state dict 中，严格加载也不会恢复它们。`original_inv_freq` 还可能保留 meta 引用。
- CPU 小型 GemmaRotaryEmbedding 复现：state dict 为空；正常频率为 `[1,.1,.01,.001]`，上述加载后是未初始化值；`original_inv_freq.device == meta`。这已确认加载模式有缺陷，未运行完整 PI05 量化其影响。
- 后续应显式重建这些 buffer，并比对普通加载与低内存加载的相同输入输出。

### P0-02：远程推理之后没有重新检查观测时效

- 位置：`real_scripts/run_ur7e_vla_safety_executor.py:338–451`；`openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py:61–65`。
- 仅在 `policy.infer` 前检查快照年龄；阻塞等待推理返回后直接使用旧 qpos、旧 OBB 计算并下发。客户端 `recv()` 没有应用层推理期限。
- 策略耗时超过 1.5 秒时，之前通过的时效检查无法保证执行时仍有效。循环与 servoJ 也没有独立的控制周期/看门狗设计。
- 后续需要执行前重新校验状态/时间、推理超时和失联停止路径。若障碍物服务与执行器分机，当前 monotonic 时间戳也不能直接跨主机相减；两者同在测试端则没有这个跨机问题。

### P0-03：预测数值缺少有效性检查，NaN 可使障碍检测失效

- 位置：执行器读取 `actions`/`prediction`/`selected` 的位置；`real_scripts/real_cbf_qp.py:76–115`。
- 返回值检查主要是维度，未检查 action/预测点有限性、索引范围与唯一性、H 一致性、OBB 数值及轴正交性。
- 已复现：有限当前点 + 全 NaN 未来点，`select_point_flow_constraints` 返回空约束。有限 nominal action 会进入无约束分支。预测异常并不总会 fail closed。
- 后续在边界统一校验，并拒绝错误状态快照；当前 `_snapshot` 也不检查障碍服务 `status=error`。

### P0-04：OBB 约束按最早危险帧去重，漏掉后续更深侵入

- 位置：`real_scripts/real_cbf_qp.py:95–115`。
- 同一点/同一盒子首次触发后加入 `seen`，后续时刻直接跳过，最后排序不能补回遗漏。
- 已复现：盒子半尺寸 .05，预测 x=.07 后 x=0；触发距离 .025 时只保留第 0 帧 `h=.02`，遗漏下一帧 `h=-.05`。
- 后续应保留最危险时刻或多时刻约束。现在保存的 `time_id` 并未参与 QP 时间尺度推导。

### P0-05：右工具几何、配置和在线状态互相矛盾

- 位置：`configs/ur7e_dual_quest3.yaml` 的 tools；`real_scripts/dual_ur7e_surface.py:63–142`；`real_scripts/live_dual_ur7e_obstacle_model.py:124–125,190,416–445,495`。
- 配置和 README 写右 EPick，实际采样器左右都调用 2F-85，障碍物剔除也将右侧当 2F-85 并连接其 GET POS。
- 即使实物已更换为双 2F-85，快照仍固定 `gripper_position.right=0.0`，没有输出读到的右夹爪角度；在线剔除与策略/CBF 的右手点云不一致。
- 后续先以真实硬件配置统一工具类型、TCP、网格、状态和动作接口；不是单纯改名即可修复。

### P1-01：CoTracker 双臂训练主线缺失

- 位置：`scripts/preprocess_quest3_hdf5.py:251–267,310–312`；`real_scripts/preprocess_left_vr_hdf5_cotracker_observations.py`。
- 默认所有表面监督由 `DualUR7eSurfacePointSampler.link_points` 得到，不是 RGB-D 实测点。CoTracker 脚本为左臂路径；合并代码明确拒绝 `arm_count != 1`，包括补成 14D 的单臂数据。
- CoTracker 本身给出图像轨迹，3D 坐标在本项目中通过深度抬升，FK 只应作为筛选辅助；这条思路存在，但尚未组成双臂主线。
- `--point-target visual` 虽可训练，但服务/执行器仍输入 FK 网格点，且 checkpoint 仍保存网格 identity/hash，无法证明 tracker ID 与网格索引相对应。当前视觉监督模型不能直接沿用该在线点输入及 FK Jacobian。
- 后续必须同时打通：双臂 tracker IDs、可见性 mask、训练/在线当前点来源，以及实测点到机器人 link/Jacobian 的对应关系。

### P1-02：主线点模块架构和训练目标不是要求的 flow matching

- 位置：`openpi/src/openpi/models_pytorch/pi0_pytorch.py:481–680`。
- `embed_safety_prefix` 把当前点和关节 token 拼进 VLM 输入；`_predict_surface_offsets` 将 action expert 的 suffix 特征与点/关节投影相加，再用 MLP 直接回归位移。
- 点监督是 `(point_offsets-target_point_offsets)^2`，没有点噪声状态、点速度目标和独立 point DiT。
- `sample_actions_and_point_offsets` 的 Euler 循环只积分 action；结束后只计算一次点位移。物理位移与生成过程中的 flow velocity 是不同量，不能因变量名有 flow 就认为满足方法。
- 旧 `safety_module/safety_flow_point_model.py:364–504` 具备 prefix/当前点条件、速度训练和 Euler 积分；但其训练器读取预缓存 prefix，仅优化 SafetyFlowPointModel，未联合训练 action expert。该模块值得复用评估，不应先删除。

### P1-03：默认训练冻结策略错误，README 已落后于代码

- 位置：`scripts/train_pi05_ur7e_surface_pytorch.py:80,493–506`；README 训练章节。
- `--freeze-base` 当前实际只冻结 PaliGemma，保留 action expert、action/time heads 和 surface heads，符合用户冻结要求。
- 但默认不传此参数，普通训练和 A100/H100 示例会训练 VLM，违背“只冻结 VLM”。
- README 的 V100 段落声称只训练新增点层，与当前代码不符；因此该命令在 16GB 卡上的可运行性不能依据旧显存说明认定。

### P1-04：front/side 没有进入当前在线障碍主入口

- 位置：`real_scripts/live_dual_ur7e_obstacle_model.py:113–115,407,455`。
- 相机列表只构造 front，LingBot 也只处理 front；缺 side 采集、标定、同步及融合。共享 LingBot helper 的默认 front/side 不改变主入口覆盖后的单视角行为。
- 实際顺序为原始深度生成机器人排除掩码 → LingBot 修复完整深度 → 原始支持域掩码筛选 → 点云/聚类。它不是修复后重新对双臂进行完整几何过滤；超过默认 2 像素原始支持域的补全点也被丢弃。
- 后续应明确支持域限制是否保留，并实现双视角统一坐标系；不能只将参数中的 camera names 改成两个名字。

### P1-05：动作与点序列仅索引对齐，数据保留和时间契约不完整

- 位置：`scripts/preprocess_quest3_hdf5.py:132,224,262–307`；外部采集器 `../quest3_collect/quest3_robot_data/recorder.py:114–115`。
- action[t] 配 surface[t+1] 的索引正确。采集器明确 commanded 为 `target_q-actual_q`，executed 为 `qpos[t+1]-qpos[t]`。默认以 commanded 为动作标签、实际下一帧几何为点标签，不保证指令完全实现，需保留此语义而不能当作相同数值。
- 只输出有完整 H 步的有效窗口，末尾 H 帧等不成为训练样本；若“每帧都有 chunk”是严格要求，需要 padding+mask 或明确例外。
- NPZ 不是原轨迹数据集的完整副本：只保存选定 RGB、选定动作源及部分元数据，未保留原深度、两种动作和 source_timestamps 等全部字段。原 HDF5 仍存在，但 source_episode 为绝对路径，迁移后关联可能失效。
- 时间戳存在，训练/控制却没有利用采样间隔建立动作时长契约；执行默认 5Hz，与采集动作周期是否一致尚待实测。

### P1-06：单臂补位数据的点契约有内部错误

- 位置：预处理 `point_ids`（第 301 行）；`scripts/validate_project_pipeline.py:59–80`。
- 补位模式 `arm_count=2`、`surface_arm_count=1`，`fixed_link_points` 只有左臂，但 `point_ids=sampler.point_ids(arm_count)` 返回双臂 ID。
- 验证器用 `model_hash(arm_count)` 检查左臂 surface hash，会把这种 README 推荐的合法补位数据报为错误。
- 后续区分动作臂数和表面监督臂数；补位不是双臂真实数据，不能代替双臂训练验收。

### P1-07：CBF 是启发式局部投影，未证明首动作修正后未来轨迹安全

- 位置：`real_scripts/real_cbf_qp.py:101–108,172–231`。
- 当前 q 上的 FK Jacobian 与未来预测最小 barrier 混用；OBB normal 从当前/未来归一化坐标最大轴选取，并不总是所用 signed distance 的梯度。跨盒另一侧等情况下需要专门测试。
- 零梯度约束直接跳过；若其他约束可解，结果仍可能 success，遗漏原本无法满足的危险点。
- 只检查离散未来采样点，不检查帧间扫掠；只修正 actions[0] 的关节量，未重新积分/预测验证修正后的未来轨迹。不能将 solver success 当作完整未来安全证明。
- 后续应明确 CBF 离散时间推导、不可控约束处理和修正后检验。右夹爪未执行；左夹爪为可选独立命令，也不在 QP 变量中。

### P2：其他完整性问题

- 障碍物时间门控要求连续命中 3 帧，新障碍前两帧不进入安全快照；漏检一帧就不输出该盒子（`TemporalOBBGate`）。显示去闪烁与控制用保守障碍集合应区分。
- 桌面候选默认高度 >=2cm，且必须靠近桌面；扁平/悬空物体会被筛掉。若任务只允许指定桌面物体，可保留，但需写入检测范围契约。
- 训练加载器只要求存在 `coordinate_frame`，未验证值，保存 metadata 却固定为 left_base；`--validate-only` 的 GPU/显存检查发生在退出前，可能阻塞本应纯数据检查的用途。
- 在线右 gripper 未在 finally 关闭；读取失败保留旧角度且不标记状态过期。
- 相机分辨率不是 1280×720 时只警告，仍需核对内参缩放，不能视为已支持任意尺寸。

## 3. 冗余与归档候选（本轮不删除）

| 范围 | 判断及后续处置 |
| --- | --- |
| 训练/服务两个 `_Tokenizer`、缓存查找、MODEL_IMAGE_KEYS | 确定存在重复契约逻辑，宜抽取共享模块，避免分叉 |
| `live_dual_ur7e_obstacle_model.py` 的 `tool_proxy_points` / `box_surface` | 当前源码引用仅停留在函数间，无主流程调用；可列删除候选 |
| 上述文件及 `dual_ur7e_surface.py` 导入的 `sample_collision_surface` | 当前未使用导入 |
| live 的 `--left/right-tool-urdf`、`--tool-margin-m` | URDF 只检查存在/打印，实际点由硬编码 vendor helper 生成；margin 未进入实际该点生成路径。属于失效接口，应修复或删除 |
| live 工作空间裁剪后再次用同样范围计算 `local` | 重复过滤，可简化 |
| `SafetyPointDecoder`、旧 prefix 服务、LIBERO 数据生成/评估脚本 | 脱离新主线，但仍相互引用；只能整组归档，不能称为全局死代码 |
| `SafetyFlowPointModel` 与旧训练器 | 方法上更接近需求，优先评估复用，不能与旧直接回归 decoder 一起删除 |
| `scripts/*libero*`、`*robosuite*`、`*upright_blocks*`、SDF/safe-space 工具 | 仿真实验支线，建议迁往 experiments；若保留基准评测则有用 |
| `ur7e_inference/`、内置遥操 collector、legacy left policy | 与 Quest3/当前执行器并存的历史入口，建议单列 legacy/采集工具；需先确认运维用途 |
| 多个重建/标定/可视化 demo | 存在职责重叠，不等于无用；按调用关系和日常硬件维护需求合并 |
| `libsurvive/`、Pika SDK、robot vendor assets、LingBot/OpenPI | 第三方或硬件依赖，不能仅因主入口未直接 import 就删除 |

## 4. 名称不一致记录

| 目标概念 | 当前名称/问题 | 后续方向 |
| --- | --- | --- |
| VLAsafe | README 为 Dual UR7e Safety World-Action Model | 以surface action model进行方法命名 |
| pi05 backbone | PI0Pytorch / PI05SafetyPytorch / pi05_base 混用 | pi05 backbone指代在此pi05模型上改造； PI0Pytorch / pi05_base是openpi的官方名称，不作修改；PI05SafetyPytorch应该是本方法的名称，可以修改为SurfaceActionModelPytorch|
| point DiT / point decoder | 主线 `surface_offset_head`；另有 SafetyPointDecoder、SafetyFlowPointModel | 先统一架构，再规范名称，避免将 MLP 改名成 DiT(会统一修改为DiT结构) |
| surface_chunk | fixed_link_points、current_link_points、target_point_offsets、predicted_robot_points | current_surface指代当前帧surface、surface_chunk是未来一系列帧的surface，长度与action_chunk对应、其他先不做修改 |
| action_chunk | action_chunks / actions | 数据集复数与单次返回可合理不同，但必须有 schema |
| live_obstacle_detection | live_dual_ur7e_obstacle_model.py | 明确公共入口；当前不是用户要求的双视角实现，统一名称为live_obstacle_detection，可视化为live_obstacle_detection_viewer |
| robot surface | arm_points、link_points、robot_points、visual_robot_* | 区分实测点、网格点、link-local 点及坐标系，本逻辑仅使用surface_points名称指代机械臂的表面点 |
| right EPick / 2F-85 | config、README、point_identity_version、模型代码冲突 | 属于几何语义错误，优先于命名整理 |
| front/side 与 D435i/D455 | generic D435i class 服务 D455；大量历史文件名 | 型号复用本身不必然有错，但相机角色与型号需分离，一共四个相机，front和side为d455，两个wrist为d435i |
| `left_to_camera` 等 | 实际 matrix 在注释中为 camera→left_base | 使用 T_left_base_from_front 等方向明确的变量名 |

命名迁移需更新 checkpoint/schema 版本，保留兼容转换，不应只搜索替换。

## 5. 路径与环境设计

1. **缺根级可复现安装定义**：README 声称统一 conda safety，但没有提供完整根级 environment/依赖锁。OpenPI 清单不覆盖所有 h5py、SciPy、trimesh、Plotly、RTDE、RealSense、CoTracker、LingBot 用途。建议分别定义 preprocess、policy、runtime 环境和固定版本，列出系统驱动依赖。
2. **conda 与 uv 补丁位置不一致**：README 用 cp 覆盖 `$CONDA_PREFIX/lib/python3.11/site-packages/transformers`，正式训练却用 `uv run --project openpi`。独立 uv 环境不会因 conda 被打补丁而自动包含补丁；重装 transformers 也会覆盖手工修改。应使用可重复安装的固定补丁包/构建流程并验证实际解释器。
3. **CoTracker 个人路径硬编码**：`preprocess_left_vr_hdf5_cotracker_observations.py:31–32` 指向 `/home/mypc/.cache/...`，并在跟踪中固定使用 CUDA。应统一资产配置、缓存根与 device 参数。
4. **主线强依赖相邻仓库**：采样器构造就加载 `../quest3_collect/config/hardware_dual_teleop.yaml` 及标定，即使只处理单臂数据。拷贝 safety 仓库无法独立复现。可保留外部采集数据路径，但标定/硬件版本应作为显式部署资产提供。
5. **配置未真正集中**：model/pretrained、相机映射、策略地址等在 YAML、脚本默认和 metadata 各有来源；训练器也不读取根项目 config。加载新 project config 后部分路径仍按 REPO_ROOT 解析，应明确该规则。
6. **策略网络默认错误**：执行器 `--policy-host=127.0.0.1`；旧 runtime 为 `192.168.125.59`。目标应配置为 `192.168.124.64`。服务监听 0.0.0.0 本身可用于远程，不是阻塞点。
7. **旧安装文档不对应当前仓库**：`ur7e_inference/docs/environment_setup.md` 使用 Sci-VLA/third_party/openpi、Windows 个人目录和旧 IP；不能作为当前 Linux 双臂主线安装指南。
8. **模型依赖来源不统一**：仓内有 LingBot，却在 import 错误提示中建议从未固定 commit 的远程 Git 安装；requirements 多数不锁版本。应明确使用仓内源码还是固定发布版本。此结论来自本地文件，未在线核验新版本兼容性。
9. **脚本与库混排**：大量 sys.path 注入，真实硬件类、模型、demo、实验入口混在 scripts/real_scripts，增加导入副作用与依赖耦合。建议建立可安装的 vlasafe 包，CLI/实验/第三方分开，先迁移再删旧入口。
10. **数据存储扩展性**：重叠 H 窗口保存 current/offsets 等重复数组，训练器把所有 shard 和 RGB 全量载入内存；DDP 每进程重复加载。大规模双臂+双视角数据应考虑按帧存储、索引窗口和惰性读取。
11. **测试发现配置断裂**：根 `.gitignore` 忽略 tests/；OpenPI pytest testpaths 不包含根 verification。`verification` 测试还依赖外部标定路径。应集中测试入口并将纯几何测试改为自包含 fixture。

建议目录边界（供后续设计，不在本轮实施）：`src/vlasafe/{data,models,perception,safety,runtime}`、`configs/{hardware,training,deployment}`、`scripts/` 薄 CLI、`tests/`、`experiments/legacy/`、`third_party/`、`outputs/`。

## 6. 验证结果与后续验收

- 已运行：`PYTHONPATH=openpi/src:openpi/packages/openpi-client/src:. conda run -n safety python -m pytest verification/test_dual_ur7e_pipeline.py -q`，**5 passed**。
- 168 个 Python 文件 AST 解析通过，不代表运行时依赖完整或行为正确。
- 小型复现确认：meta 加载 RoPE 未恢复、NaN 预测产生零 OBB 约束、多时刻去重漏掉更深侵入。
- 现有 QP 测试只断言 shape/triggered，没有断言 success、约束满足或修正后安全，不能覆盖以上缺陷。
- 未运行：真实双视角 CoTracker 预处理、完整 checkpoint 前向、实际训练、跨机策略请求、相机/机器人控制。没有启动任何物理动作。

后续顺序建议：先处理 P0 加载与运行边界问题，统一真实工具/数据契约；再接通双臂实测点、独立 point flow 模块及默认冻结策略；随后完成双视角在线链路与首动作闭环验收；最后做命名迁移、历史代码归档与环境重构。

必要验收至少应包括：真实双臂 HDF5→实测点窗口对齐；VLM 无梯度而 action expert/point module 有梯度；点速度监督与多步积分；正常/低内存 checkpoint 加载一致；双相机标定融合与机器人剔除；预测异常/网络超时/陈旧障碍 fail closed；QP 最危险时刻与零梯度约束处理；动作执行前状态校验及离线回放。应先完成离线验收，再做受控实机验证。
