# oavla

**oavla** 是面向双 UR7e + Robotiq 2F-85 的四视角视觉语言动作模型，同时输出动作序列和 front/side 配对旋转框（OBB）。模型已独立封装，可脱离 OpenPI 使用；本仓库还提供数据标注、训练、socket 推理，以及基于 Voxblox 和椭球建模的本地避障客户端。

| 能力 | 输入与输出 |
| --- | --- |
| 动作预测 | `front / side / right_wrist / left_wrist` 四路 RGB、机器人状态和文本指令 → 动作序列 |
| 障碍物检测 | 仅使用 front/side 图像及指令、状态 → 两视角旋转 OBB；检测分支不读取腕部图像或其融合特征 |
| 对象配对 | 一帧可有多个对象，同一对象在两个视角中共享 `pair_id`，无需类别名称 |
| 本地避障 | 共享 front/side RGB-D → Voxblox 体素 → 配对 OBB 筛选 → 障碍椭球 → MCBF/QP 动作优化 |

模型不输入或输出点云。深度仅用于本地避障建图，不发送给模型。`pair_id` 只建立当前预测内的跨视角对应关系，不表示跨帧跟踪 ID。

项目展示名称统一为 **oavla**；现有目录、安装包和命令沿用 `oa-vla`，Python 导入名为 `oa_vla`。下列命令除特别说明外，均在仓库根目录执行。

## 快速开始：本机 socket 测试

需要 Python 3.11 或 3.12。在推理环境安装独立模型：

```bash
python -m pip install -e './oa-vla[serve]'
```

没有训练权重时，可创建 tiny 随机 checkpoint，验证模型加载和通信链路。在终端一执行：

```bash
oa-vla-init-test --output outputs/oavla_demo/policy
OMP_NUM_THREADS=1 oa-vla-serve \
  --model outputs/oavla_demo/policy --device cpu --num-steps 1 \
  --host 127.0.0.1 --port 8000
```

初始化命令要求输出目录尚不存在。服务默认监听 `127.0.0.1:8000`，不打开相机或机器人。
在终端二准备四张 RGB 图片，发送一次请求：

```bash
oa-vla-send --host 127.0.0.1 --port 8000 \
  --front /data/front.png --side /data/side.png \
  --right-wrist /data/right_wrist.png --left-wrist /data/left_wrist.png \
  --state 0 0 0 0 0 0 0 0 0 0 0 0 --prompt 'avoid the obstacle' \
  --output outputs/oavla_demo/reply.json
```

将图片路径替换为实际文件。示例状态为默认 12 维关节状态；正式模型须使用与训练一致的状态定义。
回复包含服务元数据、请求 context 和 `prediction`。测试模型会返回动作张量及空 OBB，标记
`test_only=true`、`detection_status=not_trained`，用于通信与形状验证；本地硬件客户端会拒绝执行该模型。

已有正式训练导出时，改用：

```bash
oa-vla-serve --model /models/oavla --device cuda --host 127.0.0.1 --port 8000
```

完整协议、动作语义与部署参数见 [socket 与本地避障指南](docs/OA_VLA_SOCKET_RUNTIME.md)。

## 数据标注与训练

先安装数据读取依赖，再启动双视角标注工具：

```bash
python -m pip install -e './oa-vla[data,serve]'
oa-vla-annotate --dataset /data/recordings --port 8765
```

打开 <http://127.0.0.1:8765>，为同一帧 front/side 中的同一对象使用相同编号，分别绘制并旋转框。
每个视角可标多个对象；某侧不可见需显式标记。“已审核无障碍”与“未标注”分开保存，标签直接写入源 HDF5。

训练数据采用包含四路 RGB、状态和动作序列的 `.npz` 或 `.shard`，支持内嵌配对 OBB 标签，
也可引用源 HDF5 标注。字段、坐标约定及可见性规则见 [独立模型说明](oa-vla/README.md#训练)。

```bash
oa-vla-train --dataset /data/rgb_action --output /models/oavla --validate-only

oa-vla-train --dataset /data/rgb_action --output /models/oavla \
  --pretrained /weights/pi05_base/model.safetensors \
  --sam3-checkpoint /weights/sam3.pt \
  --tokenizer-model /weights/tokenizer.model \
  --device cuda --precision bfloat16
```

模型以 PI0.5 风格 action expert 预测动作，使用 SAM 3 检测解码器、跨视角对象 query 融合与旋转框头预测配对 OBB。
基础权重与 tokenizer 需自行准备；原生检测权重只初始化 decoder/scorer，新增适配层、配对模块和旋转框头仍需训练。
可用 `--stage action` / `--stage detection` 分阶段训练，用 `--resume` 继续训练。

训练导出包含 `config.json`、`model.safetensors`、`metadata.json`、`processor.json` 和 tokenizer，
同时保存归一化参数、检测训练状态、动作布局、来源与周期。旧版权重需先按 [迁移说明](oa-vla/README.md#转换之前的模型) 转换，再训练新增模块。

## 本地相机与避障执行

推理服务与本地感知/控制可使用独立环境。本地客户端统一采集四路相机：四路 RGB 发往 socket，
复制的 front/side RGB-D 留在本机持续建图，避免多个进程争用相机。两个腕部相机仅开启 RGB。

在本地控制环境安装依赖并构建 Voxblox：

```bash
cd requirements
python -m pip install -r oa_vla_runtime.txt
cd ..
python -m pip install --no-deps -e ./oa-vla
python scripts/build_voxblox.py
```

Voxblox 还需要 CMake、C++ 编译器、Eigen3、glog 和 Protobuf，详见 [运行环境](docs/OA_VLA_SOCKET_RUNTIME.md#环境)。
相机序列号、机器人配置与标定来自 [项目配置](configs/ur7e_dual_quest3.yaml) 和 `configs/hardware/calibration/`。

检查配置和依赖，此步骤不连接设备：

```bash
python scripts/run_oa_vla_avoidance.py --prompt 'avoid the obstacle' --check-only
```

正式模型服务启动后，运行本地只读采集与规划：

```bash
python scripts/run_oa_vla_avoidance.py --host 127.0.0.1 --port 8000 \
  --prompt 'move the instrument around the obstacle' --max-cycles 10
```

默认不下发动作；加 `--execute` 后将优化结果发送给机械臂和夹爪。每次只使用当前响应的第一条动作执行一个有界控制步，
随后重新观测。框筛选使用该请求对应的 RGB-D 和 TSDF 快照：同对象两视角取交集，不同对象取并集。
同一对象在双视角框内没有占据体素时，自动按透明障碍处理：将两个框反投影为视锥，
在工作空间边界内求交并生成包围椭球，直接参与动作优化。该补全同时用于推理避障和标注回放，
不写入实测体素地图。推理时，双视角都有框但空间无交集的配对按低置信检测忽略，
该帧回退到完整实测体素约束，并保留其他有效透明椭球；全部配对被忽略且无实测障碍时，允许动作进入正常控制流程。
空检测、单侧不可见、过期响应仍进入 hold。人工标注回放对无交集配对仍报错。
输出目录保存 `report.json`、`steps.jsonl`，其中 `paired_geometry` 区分实测与透明障碍。

当前约束覆盖已建模障碍和左右臂之间的距离，不包含同臂自碰撞或连续轨迹扫掠检查；覆盖范围依赖检测、深度与标定。
执行规则、参数及限制见 [完整指南](docs/OA_VLA_SOCKET_RUNTIME.md#执行与时效规则)。

## 代码与文档入口

| 入口 | 用途 |
| --- | --- |
| [oa-vla/](oa-vla/README.md) | 可独立安装的模型、标注、训练、权重迁移和推理 API |
| [serve_oa_vla.py](scripts/serve_oa_vla.py) | 仓库内 socket 服务入口，等效于 `oa-vla-serve` |
| [send_oa_vla_input.py](scripts/send_oa_vla_input.py) | 四张 RGB 文件与状态、指令的单次发送入口 |
| [run_oa_vla_avoidance.py](scripts/run_oa_vla_avoidance.py) | 共享相机采集、建图、动作优化及硬件执行入口 |
| [socket 与本地避障](docs/OA_VLA_SOCKET_RUNTIME.md) | 协议、环境、部署、时效与执行规则 |
| [标定](docs/CALIBRATION.md) · [单臂 Pika](docs/SINGLE_UR7E_PIKA.md) | 相机、双臂和 TCP 标定 |
| [实时感知](real_scripts/README.md) · [模型无关回放](docs/MODEL_FREE_OBSTACLE_VALIDATION.md) | 独立感知与录制动作回放工具 |
| [批量标注](docs/BATCH_OBSTACLE_ANNOTATION.md) · [深度与数据导出](docs/QUEST3_DEPTH_ALIGNMENT.md) | 父项目的数据工具 |

旧集成版训练、WebSocket 服务和点云模型入口保留在 [历史文档](docs/legacy/INDEX.md) 及
[集成版说明](docs/PI05_DETECTION_EXPERT.md) 中。当前独立模型使用 format/schema 2 和 TCP MessagePack 协议，
请以本页及独立模型 README 的命令为准。
