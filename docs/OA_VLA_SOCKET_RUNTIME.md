# oavla socket 与本地避障执行

服务端只运行独立 oavla，默认 `127.0.0.1:8000`。本地客户端独占相机，四路 RGB 发给模型，
front/side RGB-D 留在本机供 Voxblox 使用。后台持续采集、读取关节反馈及建图，推理等待不占用这些线程。
硬件输出只在客户端主线程发生；默认只读采集和规划，`--execute` 开启机械臂与夹爪输出。

## 本机测试模型与产物

已有本机测试产物时，未训练测试模型位于 `outputs/oa_vla_test/policy`，是可加载的 tiny 架构及随机参数，
不是空文件，也不是正式预训练权重。服务地址与模型信息见 `outputs/oa_vla_test/server.json`，
日志见 `outputs/oa_vla_test/server.log`，一次实际 socket 响应见 `outputs/oa_vla_test/prediction.json`。
这些测试产物在 `outputs/`，不纳入源码；新环境可按下文命令重新生成。服务是否运行需实际连接确认。

测试模型的 handshake 声明 `test_only=true`、`detection_trained=false`，返回动作和空 OBB，
`detection_status=not_trained`。本地实机入口会拒绝它，避免把随机输出作为已训练的检测/动作。
体素配对、椭球和 QP 的测试使用已知合成障碍与 mock 硬件，不通过伪造训练标记驱动真实设备。

## 环境

推理环境（Python 3.11/3.12），在仓库根目录：

```bash
python -m pip install -e './oa-vla[serve]'
```

本地感知/控制环境可独立安装，不需要 OpenPI、JAX、LeRobot、PyArrow 或模型的 PyTorch/Transformers：

```bash
cd requirements
python -m pip install -r oa_vla_runtime.txt
cd ..
python -m pip install --no-deps -e ./oa-vla
python scripts/build_voxblox.py
```

`--no-deps` 在此仅用于安装轻量 socket 客户端；所需 NumPy/Pillow/MessagePack 已在 runtime requirements 中。
Voxblox 构建需要 CMake、C++ 编译器、Eigen3、glog、Protobuf，按 [环境说明](IMPLEMENTATION_GUIDE.md#环境) 准备。
默认不加载 LingBot 或 CoTracker，不启用 ESDF 和网页。

检查依赖不连接任何设备：

```bash
python scripts/run_oa_vla_avoidance.py --prompt 'avoid the obstacle' --check-only
```

## 服务端

已有训练好的 oavla format-2 导出目录时：

```bash
oa-vla-serve --model /models/oa-vla --device cuda --host 127.0.0.1 --port 8000
# 等效：python scripts/serve_oa_vla.py ...
```

远端 GPU 主机部署时显式指定相应私网绑定地址，并在客户端指定 `--host`。
服务按连接执行有序请求，多个连接共享一把模型推理锁。协议为长度前缀＋MessagePack 的 TCP，
**不兼容**旧 `serve_quest3_pi05_safety.py` 的 OpenPI WebSocket 协议；旧服务不负责新模型。

训练导出现在记录 `action_semantics`、`action_layout`、`action_source` 和 `action_period_s`。
缺少这些字段的旧导出，只有在确认与训练数据一致后，才显式提供：

```bash
oa-vla-serve --model /models/oa-vla --device cuda \
  --action-semantics interleaved_joint_delta_gripper_target \
  --action-source commanded --action-period 0.05
```

已有元数据与这些覆盖值冲突会拒绝启动。未知动作布局不能仅凭 14 维猜测。

没有权重时生成测试模型，再以前台方式启动（端口已占用时先停止现有测试服务或选其他端口）：

```bash
oa-vla-init-test --output outputs/oa_vla_test_new/policy
OMP_NUM_THREADS=1 oa-vla-serve --model outputs/oa_vla_test_new/policy --device cpu --num-steps 1
```

## 单次发送四张图片

```bash
python scripts/send_oa_vla_input.py --host 127.0.0.1 --port 8000 \
  --front outputs/oa_vla_test/inputs/front.png \
  --side outputs/oa_vla_test/inputs/side.png \
  --right-wrist outputs/oa_vla_test/inputs/right_wrist.png \
  --left-wrist outputs/oa_vla_test/inputs/left_wrist.png \
  --state 0 0 0 0 0 0 0 0 0 0 0 0 --prompt 'avoid the synthetic obstacle' \
  --output outputs/oa_vla_test/reply.json
```

也可使用 `oa-vla-send` 或 `python -m oa_vla.send`。此入口不连接相机或机器人。
传输默认使用原始 uint8 字节，可选 `--encoding png` 或 `jpeg`；不使用 JSON/base64 传大数组。
消息上限 32 MiB。模型在内部 letterbox，返回的 OBB 始终使用请求图像的原始像素坐标。

每次请求有 session、sequence、request_id、frame_id、各相机采集时间戳/帧号/尺寸、标定 ID 和状态时间戳。
服务原样回传 context；客户端拒绝响应错配、重复和乱序，不会在断线后自动重发动作。
跨机器的时间戳只原样回传，时效检查使用采集客户端自己的 monotonic 时钟。

## 本地实时避障

使用正式已训练的模型服务时，先在客户端环境运行只读模式：

```bash
python scripts/run_oa_vla_avoidance.py --host 127.0.0.1 --port 8000 \
  --prompt 'move the instrument around the obstacle' --max-cycles 10
```

默认自动启动四路采集、Voxblox 和椭球建模。不会恢复到录制初始姿态，不需要人工标注文件、
`selection.json`、旧地图或另起一个相机进程。相机序列号、双臂外参及 TCP 来自项目硬件配置。
front/side 开启 RGB-D 并对齐深度到 RGB，两个 wrist 只开启 RGB，不占用额外的腕部深度带宽。

同一命令加 `--execute` 启用实机：

```bash
python scripts/run_oa_vla_avoidance.py --host 127.0.0.1 --port 8000 \
  --prompt 'move the instrument around the obstacle' --execute
```

运行目录包含 `report.json` 和 `steps.jsonl`，记录请求对应帧、原动作目标、优化后目标、
每个对象的体素数、QP/距离松弛信息和执行/hold 原因。`--max-runtime` 限制运行秒数，Ctrl+C 停止并释放资源。
只读模式仍采集真实相机和反馈、执行真实建模及规划，但不创建 RTDEControl 或夹爪写接口。

## 执行与时效规则

1. 相机 SDK 只由一个线程启动/读取/停止。RGB-D 进入共享缓存时复制并转为只读数组，推理和建图随后复用同一帧；
   最新帧槽位覆盖旧数据，避免队列积压。每个推理请求固定引用自己的 RGB-D 和 TSDF 快照。
2. Voxblox 在固定、有限的双臂工作空间持续更新，不用上一帧框裁剪地图边界。
   返回框只筛选同一帧的快照：同对象两个视角取交集，不同对象取并集；逐对象统计实际占据体素。
   某个对象无占据体素时，自动用双视角框的有界视锥交集生成透明障碍包围椭球，直接参与 MCBF/QP。
3. 每次响应只执行第一条动作的一个有界控制 tick。执行完成/明确 hold 后，下一次推理必须使用之后捕获的新帧。
   其余 horizon 不缓存执行，不累加未来 commanded deltas。
4. 增量布局为 `[Δq_left(6),g_left,Δq_right(6),g_right]`，先转换为
   `[q_observation + Δq_left/right,g_left,g_right]`，再由当前反馈下的 MCBF 修正。
   `absolute_q12_gripper2` 也受显式协议支持。夹爪只做限速，不作为 QP 变量。
5. 下发前后检查反馈有效性，规划后再次检查帧年龄与关节漂移。默认单次关节步长最多 0.015 rad，
   同时受速度、加速度限制。控制周期必须不大于训练动作周期；例如训练周期 0.008 s 时需要 `--control-period 0.008`。
6. 过期、缺框、单侧不可见、空检测、越界动作、不可接受的松弛/预测 barrier 都进入 hold，
   停止当前输出并重新观测。采集/反馈/socket 故障终止运行并清理资源。
   测试或未训练模型在打开运动接口前就会被拒绝。双视角都有框但空间无交集的对象按下述低置信配对规则处理，不因此 hold。

常用参数：`--max-result-age`（默认 0.75 s）、`--max-state-drift`（0.01 rad）、
`--max-planning-drift`（0.001 rad）、`--max-feedback-age`（0.15 s）、
`--max-distance-slack`（1e-5 m）、`--voxel-resolution`（0.02 m）、`--no-robot-mask`。
默认开启 robot mask；TSDF/QP/推理的实际时延需要在目标机器和分辨率下测量，不能把频率配置当作性能保证。

### 无体素区域的透明障碍补全

此规则默认启用，与标注回放共用实现，无需额外开关。它按同一个 `pair_id` 分别判断，
所以一个对象有体素不会掩盖另一个对象缺少深度支持。若后续帧重新出现占据体素，自动恢复实测几何。

补全使用标定后的 front/side RGB 内外参、原图旋转框及工作空间网格的有向边界，
将两个视锥在相机光轴深度 0.01–2.5 m 内求交，再拟合覆盖该区域顶点的椭球。
针孔相机使用连续几何，不要求区域中存在体素中心；有畸变时使用采样边界加角度余量近似外包。
格式无效、单侧缺框、失效的采集/标定仍然拒绝，不会凭单个框推断透明物体。
双视角都有框但无三维交集时，推理客户端丢弃该低置信配对；人工审核标注的回放入口仍报错。

`obbs` 中补全项包含 `geometry_source=dual_rgb_transparent_region`、`object_id`、
`ellipsoid_shape`、`region_vertices`、`voxel_count=0` 和 `depth_measured=false`。
它们保留为独立椭球，随实测障碍一起进入优化；兼容的 OBB 包络供其他框约束使用。
实测 `occupancy_map` 的状态和占据数不会被伪造。快照诊断 `paired_rgb_obstacles`、
执行日志/报告的 `paired_geometry` 记录每个对象的体素数、透明补全及忽略的对象。

### 双视角无交集：忽略低置信配对

推理避障对“只有一侧有框”和“两侧都有框但无交集”采用不同规则：

| 证据 | 推理避障行为 |
| --- | --- |
| 只有一侧有该对象的框 | 可能遮挡，进入 hold 并重新观测 |
| 两侧有框且有占据体素支持 | 按实测体素建模 |
| 两侧有框、有空间交集、无占据体素 | 透明障碍包围椭球参与优化 |
| 两侧有框但在限定空间内无交集 | 记录并忽略该配对，不生成透明障碍，不因该配对停止动作 |

出现无交集配对时，该帧回退到**完整工作空间的实测占据体素**，不让不可靠的框裁掉真实障碍；
其他仍有效的透明椭球也保留。若所有配对均被忽略，且实测地图无占据体素，则允许动作进入正常控制流程，
仍执行双臂距离、关节范围、速度/加速度和时效检查。格式错误、坏标定等异常不会按无交集配对忽略。

诊断 `ignored_objects` / `ignored_count` 记录被忽略的对象，逐对象标记
`geometry_source=ignored_inconsistent_rgb_pair`、`reason=no_3d_intersection`；
`measured_voxel_scope=full_workspace_on_inconsistent_pairs` 表示已回退到完整实测地图。
回放使用人工审核标注，不自动忽略无交集对象，仍要求修正标注或标定。

这里的“透明”是按双视角 RGB 和缺少占据体素作出的避障假设，不能确认物理材质或精确厚度。
深度范围只由双视角几何和工作空间边界约束，椭球可能偏大，也可能覆盖两工作空间之间的间隙。

现有椭球控制器包含双臂对环境和左右臂之间的约束，不包含同臂自碰撞；本入口额外检查所有已建约束的
预测 barrier 和松弛，但尚未增加连续轨迹扫掠检查。正常框筛选时，除透明补全区域外，OBB 外未选中的体素不构成避障约束；无交集回退时使用完整实测体素，
本链路的覆盖范围受检测召回、深度有效性与标定精度限制。

## 验证

```bash
# 独立模型/协议/真实 loopback socket；受限制的 CI 可能跳过 socket 用例
OMP_NUM_THREADS=1 python -m pytest -q -c oa-vla/pyproject.toml oa-vla/tests
# 本地合成帧、配对体素、椭球/QP、mock 硬件与线程清理
OMP_NUM_THREADS=1 python -m pytest -q verification/test_oa_vla_runtime.py \
  verification/test_realsense_stream_startup.py verification/test_replay_voxblox.py
```

mock 端到端用例采用显式注入的合成相机/占据地图/机械臂；生产入口不提供 NumPy 替代 Voxblox 的静默回退。
Voxblox 原生库缺失时相关原生测试会跳过，实机入口会明确报告缺失依赖。
