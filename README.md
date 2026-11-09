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

LIBERO-Safety 实验入口：[双卡离线训练](#libero-safety双卡离线训练lerobot) ·
[部署与独立测试](#libero-safetysocket-部署独立测试与论文指标)。

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

## LIBERO-Safety：双卡离线训练（LeRobot）

目标模型为 **π0.5、GR00T N1.7、LingBot-VLA 2.0**，统一要求使用 LeRobot 训练框架。
下方三份命令均运行 LeRobot 的 `lerobot-train`，在集群分配的**单节点两张 GPU**上执行，
每次运行一种模型；每卡 batch size 为 1，共训练 10,000 个优化步，每 2,000 步保存。
π0.5、GR00T 保留原有 LeRobot 0.6.0 命令；LingBot-VLA 2.0 使用已内置该模型的官方源码版本，
不能直接放进 0.6.0/0.6.1 的发布包环境中运行。

π0.5 使用全量微调，GR00T 冻结视觉/语言骨干、训练 projector/action head，二者使用 DDP，
每卡保留一份模型。LingBot-VLA 2.0 使用全量微调及双卡 FSDP2 分片，并保持 FP32 参数、BF16 计算。
相同步数不代表相同可训练参数量或计算量；仅知道双卡不能保证显存足够。

前两份命令按 [LeRobot v0.6.0 训练配置](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/configs/train.py)
编写。第三份命令及安装方式单独标注源码提交；不要混用新旧版本的分布式和日志参数。
关闭仿真评测均使用 `--env_eval_freq=0`。
旧 [集群指南](docs/LIBERO_SAFETY_CLUSTER_TRAINING.md) 和 `scripts/cluster/train_libero_safety_pi05.slurm`
中的 v0.4.4 / 四卡设置属于旧版流程，0.6.0 双卡训练以本节为准。

### 环境、离线资产与路径

训练使用单独的 Python 3.12+ 环境，与下节 NumPy<2 的仿真环境分开。
π0.5、GR00T 的以下命令固定 LeRobot `0.6.0`；LingBot-VLA 2.0 的独立环境按其小节安装。
已有环境应确认安装了对应 extra：π0.5 为 `training,pi`，GR00T 为 `training,groot`。
例如，在已备齐依赖 wheel 的离线节点安装 π0.5 训练环境：

```bash
python -m pip install --no-index --find-links=/data/wheelhouse \
  'lerobot[training,pi]==0.6.0'
python -c 'from importlib.metadata import version; assert version("lerobot") == "0.6.0"; print(version("lerobot"))'
```

`/data/wheelhouse` 替换为本地依赖目录；GR00T 在自己的环境中替换 extra。
新版源码中 LingBot-VLA 2.0 的 extra 是 `lingbot_vla_v2`，不能用 `lingbot_va` 代替。
以下路径沿用已有集群目录。GR00T、LingBot-VLA 2.0 和缓存目录是约定示例，执行前改成实际本地位置。

```bash
export RAW_DATASET_ROOT=/seu_nvme/home/dishimin/230259982/MyDatasets/libero_safety_raw
export DATASET_ROOT=/seu_nvme/home/dishimin/230259982/MyDatasets/libero_safety
export PI05_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/pi05_base
export GROOT_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/GR00T-N1.7-3B
export LINGBOT_VLA2_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/lingbot_vla_v2_base
export QWEN3_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/Qwen3-VL-4B-Instruct
export LINGBOT_TEACHER_DIR=/seu_nvme/home/dishimin/230259982/MyModels/lingbot-vla-v2-6b
export MOGE2_FILE=/seu_nvme/home/dishimin/230259982/MyModels/moge-2-vitb-normal/model.pt
export OUTPUT_BASE=/seu_nvme/home/dishimin/230259982/outputs

# 必须指向已预先拷入完整模型/分词器缓存的位置，不要切到空缓存。
export HF_HOME=/seu_nvme/home/dishimin/230259982/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

export LIBERO_INPUT_FEATURES='{"observation.images.image":{"type":"VISUAL","shape":[3,256,256]},"observation.images.image2":{"type":"VISUAL","shape":[3,256,256]},"observation.state":{"type":"STATE","shape":[8]}}'
export LIBERO_OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[7]}}'
```

| 模型 | 离线资产要求 |
| --- | --- |
| π0.5 | `PI05_MODEL_DIR` 是带 `type=pi05` 的 LeRobot checkpoint；还需缓存 `google/paligemma-3b-pt-224` 的 tokenizer |
| GR00T | `GROOT_MODEL_DIR` 是完整 NVIDIA `GR00T-N1.7-3B` 快照，保留权重、`config.json`、`processor_config.json`、`statistics.json` 等；还需在 `HF_HOME` 中缓存 `nvidia/Cosmos-Reason2-2B` 的 processor/tokenizer。N1.5 不适用本命令 |
| LingBot-VLA 2.0 | `LINGBOT_VLA2_MODEL_DIR` 是完整 `lerobot/lingbot_vla_v2_base` 快照，配置中 `type=lingbot_vla_v2`；Qwen3-VL-4B-Instruct 提供本地 config/processor/tokenizer。保留基础 checkpoint 的蒸馏训练还需要 MoGe-2、LingBot-Depth 和 DINO-Video teacher/config，路径在下方显式设置；不使用 Wan 组件 |

若使用 Slurm，将公共环境变量和下方**一种**训练命令放进作业脚本；资源头可使用
`#SBATCH --nodes=1`、`#SBATCH --ntasks=1`、`#SBATCH --gres=gpu:2`，分区、CPU、内存和时间按集群修改。
π0.5、GR00T 激活各自环境后运行一次 `accelerate launch`；LingBot-VLA 2.0 在其环境运行一次
`torchrun`，后者同样启动 LeRobot 训练脚本。不要启动两份相同 launcher。
保留调度器设置的 `CUDA_VISIBLE_DEVICES`。

### 数据准备：从 raw 派生 v3 数据

原始 `libero_safety_raw` 保留不动，训练指向派生目录 `libero_safety`。
仅当源数据是本项目支持的官方 v2.1 格式时执行以下一次性转换；输出目录必须尚不存在。
示例声明原始动作为 metric delta，转换为 LIBERO 归一化 OSC：xyz 除以 0.05，旋转除以 0.5，夹爪不变。
若确认源动作已经是归一化 OSC，改为 `--action-units normalized-osc`，避免重复缩放。

```bash
python scripts/normalize_libero_safety_dataset.py \
  --source "$RAW_DATASET_ROOT" --output "$DATASET_ROOT" \
  --action-units metric-delta --link-videos

python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id=local/libero_safety --root="$DATASET_ROOT" --push-to-hub=false

python scripts/compute_lerobot_quantiles.py --dataset "$DATASET_ROOT"
```

已完成规范化和 v3 转换时跳过上述转换。训练目录必须直接包含 `meta/`、`data/`、`videos/`，
特征为 `action[7]`、`observation.state[8]` 和两路 `observation.images.image/image2`，20 Hz；
`meta/stats.json` 中包含训练数据的 q01/q99。若有验证/测试划分，统计量只从训练划分计算。
相机 key 和动作单位不能只通过更改训练参数假装完成转换。

### π0.5 训练命令

```bash
export MODEL_DIR="$PI05_MODEL_DIR"
export OUTPUT_DIR="$OUTPUT_BASE/pi05_libero_safety_10k"

accelerate launch --multi_gpu --num_processes=2 --num_machines=1 \
  --mixed_precision=bf16 --dynamo_backend=no "$(command -v lerobot-train)" \
  --dataset.repo_id=local/libero_safety --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.path="$MODEL_DIR" \
  --policy.input_features="$LIBERO_INPUT_FEATURES" \
  --policy.output_features="$LIBERO_OUTPUT_FEATURES" \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.train_expert_only=false --policy.use_relative_actions=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 --policy.push_to_hub=false \
  --batch_size=1 --num_workers=4 --steps=10000 --seed=1000 \
  --env_eval_freq=0 --save_freq=2000 --log_freq=100 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name=pi05_libero_safety_10k
```

### GR00T N1.7 训练命令

原始 NVIDIA 模型使用 `--policy.type=groot --policy.base_model_path=...`，不能将没有 LeRobot
`type` 字段的原始配置传给 `--policy.path`。这里为已规范化的自定义数据选择 `new_embodiment`，
使用本数据统计量，并显式禁用原生 LIBERO checkpoint 的额外动作解码变换，避免再次改写夹爪/动作。
参见 [0.6.0 GR00T 配置](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/groot/configuration_groot.py)。

```bash
export MODEL_DIR="$GROOT_MODEL_DIR"
export OUTPUT_DIR="$OUTPUT_BASE/gr00t_libero_safety_10k"

accelerate launch --multi_gpu --num_processes=2 --num_machines=1 \
  --mixed_precision=bf16 --dynamo_backend=no "$(command -v lerobot-train)" \
  --dataset.repo_id=local/libero_safety --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=groot --policy.base_model_path="$MODEL_DIR" \
  --policy.input_features="$LIBERO_INPUT_FEATURES" \
  --policy.output_features="$LIBERO_OUTPUT_FEATURES" \
  --policy.device=cuda --policy.embodiment_tag=new_embodiment \
  --policy.action_decode_transform=none --policy.use_relative_actions=false \
  --policy.tune_llm=false --policy.tune_visual=false \
  --policy.tune_projector=true --policy.tune_diffusion_model=true --policy.tune_vlln=true \
  --policy.use_bf16=true --policy.max_steps=10000 \
  --policy.chunk_size=40 --policy.n_action_steps=40 --policy.push_to_hub=false \
  --batch_size=1 --num_workers=4 --steps=10000 --seed=1000 \
  --env_eval_freq=0 --save_freq=2000 --log_freq=100 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name=gr00t_libero_safety_10k
```

### LingBot-VLA 2.0 训练命令（官方 LeRobot 源码集成）

[LeRobot 官方文档](https://huggingface.co/docs/lerobot/lingbot_vla_v2) 已提供原生
`lingbot_vla_v2` policy、processors 和训练支持，无需本仓库另写模型适配层。
**v0.6.0/0.6.1 正式发布标签未包含它，但当前源码已包含**；不能仅凭安装包显示的版本判断某个
源码 checkout 是否具有该功能。本节固定已核对的官方提交
[`81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee`](https://github.com/huggingface/lerobot/tree/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee)
（该提交的包版本为 0.7.1），避免将浮动 main 的参数误用于旧发布包。

在联网准备机取好该提交的完整源码、匹配的依赖 wheel、`lerobot/lingbot_vla_v2_base`、Qwen3-VL
及教师资产，拷入集群。激活新的 Python 3.12+ 训练环境，在离线节点安装：

```bash
export LEROBOT_VLA2_SOURCE=/seu_nvme/home/dishimin/230259982/MyCode/lerobot_lingbot_vla_v2
test "$(git -C "$LEROBOT_VLA2_SOURCE" rev-parse HEAD)" = 81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee
python -m pip install --no-index --find-links=/data/wheelhouse \
  -e "${LEROBOT_VLA2_SOURCE}[training,lingbot_vla_v2]"
python -c 'from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config; print(LingbotVLAV2Config.__name__)'
```

这里使用 LeRobot 已移植的实现，不安装 Robbyant 独立训练仓库的 requirements。
官方集成默认采用 SDPA，无需沿用旧 LingBot-VA 的 `attn_mode=flex` 或 Wan 组件。
本节保留 base checkpoint 的 dual-query 蒸馏配置，以下代码只改教师文件位置，保留其余结构与损失权重。
`LINGBOT_TEACHER_DIR` 中应含 `depth/model.pt`、`dino_video/teacher_step_10000.pth`、
`dino_video/config.yaml`；`MOGE2_FILE` 指向另行准备的 MoGe-2 权重。

```bash
export LINGBOT_ALIGN_PARAMS="$(python - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["LINGBOT_VLA2_MODEL_DIR"])
config = json.loads((model_dir / "config.json").read_text())
assert config.get("type") == "lingbot_vla_v2", "需要 LeRobot 格式的 LingBot-VLA 2.0 checkpoint"
params = config.get("align_params")
assert isinstance(params, dict) and params, "需要带完整蒸馏配置的 base checkpoint"
root = Path(os.environ["LINGBOT_TEACHER_DIR"])
files = {
    ("depth", "moge_path"): Path(os.environ["MOGE2_FILE"]),
    ("depth", "morgbd_path"): root / "depth/model.pt",
    ("video", "ckpt_path"): root / "dino_video/teacher_step_10000.pth",
    ("video", "config_path"): root / "dino_video/config.yaml",
}
for (section, key), path in files.items():
    assert path.is_file(), f"缺少离线教师文件：{path}"
    params[section][key] = str(path.resolve())
print(json.dumps(params))
PY
)"
```

同一规范化数据仍为 8 维状态、7 维动作。下面显式将状态前 6 维 EEF 位姿映射到 `end.position`，
后 2 维夹爪状态映射到 `effector.position`；动作前 6 维 OSC delta 与最后 1 维夹爪分别映射到对应槽。
其余 canonical 维度补零并掩码；槽映射只搬运数值，不改变旋转表示或动作单位。
数据已是 OSC delta，故关闭额外 state-relative 转换。

```bash
export MODEL_DIR="$LINGBOT_VLA2_MODEL_DIR"
export OUTPUT_DIR="$OUTPUT_BASE/lingbot_vla_v2_libero_safety_10k"

torchrun --standalone --nnodes=1 --nproc_per_node=2 "$(command -v lerobot-train)" \
  --dataset.repo_id=local/libero_safety --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.path="$MODEL_DIR" \
  --policy.input_features="$LIBERO_INPUT_FEATURES" \
  --policy.output_features="$LIBERO_OUTPUT_FEATURES" \
  --policy.tokenizer_path="$QWEN3_MODEL_DIR" \
  --policy.state_slots='{"end.position":{"origin_keys":[{"observation.state":{"start":0,"end":6}}]},"effector.position":{"origin_keys":[{"observation.state":{"start":6,"end":8}}]}}' \
  --policy.action_slots='{"end.position":{"origin_keys":[{"action":{"start":0,"end":6}}]},"effector.position":{"origin_keys":[{"action":{"start":6,"end":7}}]}}' \
  --policy.align_params="$LINGBOT_ALIGN_PARAMS" \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.device=cuda --policy.dtype=float32 \
  --policy.gradient_checkpointing=true --policy.train_expert_only=false \
  --policy.use_relative_actions=false --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --parallelism.dp_shard=2 --accelerator.mixed_precision=bf16 \
  --batch_size=1 --num_workers=4 --steps=10000 --seed=1000 \
  --env_eval_freq=0 --save_freq=2000 --log_freq=100 --trackio.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name=lingbot_vla_v2_libero_safety_10k
```

此处 `torchrun` 启动的是 **LeRobot 训练脚本**。新源码使用 `--parallelism.*`、
`--accelerator.*` 和 `--trackio.*`；前两份 0.6.0 命令仍使用各自版本的参数。
FP32 权重及优化器状态需要较多显存，分片后也不能保证两张 24 GB 卡足够；若改为 expert-only
或 LoRA，需作为不同微调设置记录。不要为省显存直接把 FP32 参数改成 BF16 参数更新。

三者最终产物在各自输出目录的 `checkpoints/last/pretrained_model/`；输出目录请使用新目录。
以上是官方接口及配置层面的核对，尚未在用户集群或其真实权重、数据上完成训练验证。

## LIBERO-Safety：socket 部署、独立测试与论文指标

训练对象是上节的 VLA；VLSA、VLPSA、OA-VLA 是下节评测的方法/安全层，分别记录结果。
当前共享 TCP 服务适配器支持 **原生 LeRobot π0.5** 和 **Panda OA-VLA**。
GR00T、LingBot-VLA 2.0 尚未接入本项目的安全层 socket 评测，二者仍需本项目的策略服务适配。
官方 LeRobot 已支持 LingBot-VLA 2.0 训练/推理，但本项目仍须将各自 processors、动作映射和
episode reset 接到共享 TCP 协议，不能直接套用仅支持 π0.5 的服务实现。
`lerobot-eval --env.type=libero` 的标准 LIBERO 测试也不能替代 LIBERO-Safety 的安全指标。

### 推理机器：启动模型服务

在局域网推理机器的 π0.5 环境运行（将路径换成训练导出实际位置）：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/serve_libero_safety_policy.py \
  --backend lerobot \
  --model /seu_nvme/home/dishimin/230259982/outputs/pi05_libero_safety_10k/checkpoints/last/pretrained_model \
  --normalized-osc-actions --device cuda --host 0.0.0.0 --port 8000
```

在 Panda OA-VLA 的独立服务环境运行：

```bash
python scripts/serve_libero_safety_policy.py \
  --backend oa-vla --model /models/oavla_panda \
  --device cuda --host 0.0.0.0 --port 8001
```

`/models/oavla_panda` 替换为真实的单臂训练导出，不能使用双 UR7e 权重。
未训练权重添加 `--test-only`。`--normalized-osc-actions` 确认训练数据的动作单位，不执行单位转换。
服务协议为 `oa_vla_tcp_v1`；其他现有 socket 服务需要协议转换。

### 仿真机器：三方法分别测试并汇总

仿真环境安装 [对比依赖](requirements/libero_comparison.txt)，并按
[场景准备说明](docs/LIBERO_SAFETY_LOCAL.md#环境和场景资产) 安装 LIBERO-Safety 配套 fork 和完整场景。
训练用 parquet/video 数据集不能替代仿真场景资产。先复制配置：

```bash
python -m pip install -r requirements/libero_comparison.txt
export PYTHONPATH="$PWD/oa-vla/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/libero_comparison_numba
export LIBERO_ROOT=/data/LIBERO-Safety

cp configs/libero_safety_endpoints.example.json configs/libero_safety_endpoints.local.json
cp configs/libero_safety_comparison.example.json configs/libero_safety_comparison.local.json
```

将 `endpoints.local.json` 的地址改成实际推理机器 IP/端口。示例让 VLSA/VLPSA 连接 π0.5 的 8000，
OA-VLA 连接 8001。任务清单中的套件、级别、任务、episode 初始化索引、禁碰物和可抓持物必须核对实际场景；
示例清单只提供少量 episode，不是完整论文评测集。

```bash
# 每个方法单独启动、单独保存；三个目录必须尚不存在。
for METHOD in vlsa vlpsa oavla; do
  python scripts/evaluate_libero_safety_methods.py \
    --libero-root "$LIBERO_ROOT" \
    --manifest configs/libero_safety_comparison.local.json \
    --endpoints configs/libero_safety_endpoints.local.json \
    --method "$METHOD" --max-steps 3000 --control-freq 100 \
    --output "outputs/${METHOD}_run01" || break
done

python scripts/summarize_libero_safety_methods.py \
  --runs outputs/vlsa_run01 outputs/vlpsa_run01 outputs/oavla_run01 \
  --output outputs/libero_comparison_table01
```

也可只取循环体，将 `--method` 直接设为 `vlsa`、`vlpsa` 或 `oavla` 单独运行。
汇总结果见 `outputs/libero_comparison_table01/table.csv`、`table.md`、`comparison.json`；
合并器会检查任务、协议和初始化状态是否一致。CSV 中 `SR_pct`、`SSR_pct`、`CR_pct` 为 0–100 的百分数。

| 论文列 | 统计口径 |
| --- | --- |
| SR ↑ | 成功 episode 数 / 全部请求 episode 数 |
| SSR ↑ | 成功且全过程无官方安全违规、无指定禁止接触的 episode 数 / 全部请求数 |
| CR ↓ | 任意物理子步发生机器人或抓持物与清单中禁碰物接触的 episode 数 / 全部请求数 |
| Steps ↓ | 成功 episodes 的控制 tick 均值，含 warmup；另存所有 episodes 的均值 |

Steps 是步数，100 Hz 时除以 100 才是仿真秒数；不是墙钟耗时。无成功时 Steps(success) 为 N/A。
CR 不覆盖清单外所有环境接触或同臂自碰。缺失/失败 episode 会使主要统计为空，不能作为有效论文结果。
当前对比使用共享实例标签和深度感知，VLSA/VLPSA 是适配版；不同策略源的结果同时包含策略差异。
完整方法定义、输出及适用边界见 [三方法对比指南](docs/LIBERO_SAFETY_METHOD_COMPARISON.md)。

### 无正式 checkpoint、无 LIBERO-Safety 场景时的测试

在单独的仿真/小模型测试环境安装上节对比依赖，并安装独立模型和原版 robosuite fixture：

```bash
python -m pip install -e './oa-vla[serve]' 'robosuite==1.4.0'
python scripts/smoke_libero_safety_methods.py \
  --output outputs/libero_methods_smoke_run01 --episodes 2 --steps 20
```

该命令生成并保存未训练的 tiny Panda OA-VLA checkpoint，实际运行 TCP 服务和 Panda/Lift 仿真，
独立测试三种安全层，输出 `<output>/results/table.csv`。这是链路测试，标记为 `test_only`，
**不能填作 LIBERO-Safety 论文成绩**。已有测试结果位于
`outputs/libero_methods_smoke_verified/results/`：三方法各 2 个 episode，SR/SSR/CR 均为 0，
Steps(success)=N/A，Steps(all)=22。正式成绩仍需完整场景与训练后的模型。

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

LIBERO-Safety 的单臂仿真实验见 [本地模型推理与安全层接入](docs/LIBERO_SAFETY_LOCAL.md)；
LeRobot 0.6.0 的 π0.5 / GR00T 训练命令，以及固定官方源码提交的 LingBot-VLA 2.0 训练命令，
见本页双卡离线训练一节。
官方数据下载、上传、旧版 LeRobot 0.4.4 Slurm 流程及当前 OA-VLA 联合训练见
[集群训练指南](docs/LIBERO_SAFETY_CLUSTER_TRAINING.md)。Panda profile 与双 UR7e profile 使用相同模型/协议实现，
状态和动作契约分别校验，现有双臂权重不能直接截断为单臂策略。

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
