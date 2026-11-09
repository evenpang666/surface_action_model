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

LIBERO-Safety 实验入口：[双卡离线训练](#libero-safety双卡离线训练lerobot-060) ·
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

## LIBERO-Safety：双卡离线训练（LeRobot 0.6.0）

以下三份命令分别训练 **π0.5、GR00T N1.7、LingBot-VA**。在集群已分配的**单节点两张 GPU**
上执行，每次运行一种模型；每卡 batch size 为 1，共训练 10,000 个优化步，每 2,000 步保存。
这是 DDP，每卡保留一份模型，不会把两卡显存合成一份。π0.5 使用全量微调，GR00T 冻结视觉/语言
骨干、训练 projector/action head，LingBot-VA 使用 LoRA；相同步数不代表相同可训练参数量或计算量。
显存不足时需要另行调整微调范围或使用分片方案。

本节按 [LeRobot v0.6.0 训练配置](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/configs/train.py)
编写，关闭仿真评测的参数为 `--env_eval_freq=0`。
旧 [集群指南](docs/LIBERO_SAFETY_CLUSTER_TRAINING.md) 和 `scripts/cluster/train_libero_safety_pi05.slurm`
中的 v0.4.4 / 四卡设置属于旧版流程，0.6.0 双卡训练以本节为准。

### 环境、离线资产与路径

训练使用单独的 Python 3.12+ 环境，LeRobot 固定为 `0.6.0`，与下节 NumPy<2 的仿真环境分开。
已有环境应确认安装了对应 extra：π0.5 为 `training,pi`，GR00T 为 `training,groot`，
LingBot-VA 为 `training,lingbot_va,peft`。例如，在已备齐依赖 wheel 的离线节点安装 LingBot-VA：

```bash
python -m pip install --no-index --find-links=/data/wheelhouse \
  'lerobot[training,lingbot_va,peft]==0.6.0'
python -c 'from importlib.metadata import version; assert version("lerobot") == "0.6.0"; print(version("lerobot"))'
```

`/data/wheelhouse` 替换为本地依赖目录；另外两种模型在各自环境中替换 extra。
以下路径沿用已有集群目录。GR00T、LingBot-VA 和缓存目录是约定示例，执行前改成实际本地位置。

```bash
export RAW_DATASET_ROOT=/seu_nvme/home/dishimin/230259982/MyDatasets/libero_safety_raw
export DATASET_ROOT=/seu_nvme/home/dishimin/230259982/MyDatasets/libero_safety
export PI05_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/pi05_base
export GROOT_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/GR00T-N1.7-3B
export LINGBOT_VA_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/lingbot_va_base
export WAN_MODEL_DIR=/seu_nvme/home/dishimin/230259982/MyModels/lingbot-va-base
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
| LingBot-VA | `LINGBOT_VA_MODEL_DIR` 是 `lerobot/lingbot_va_base` 的完整 LeRobot 格式快照；`WAN_MODEL_DIR` 是配套 Wan 资产目录，包含 `vae/`、`text_encoder/`、`tokenizer/`，仅 DiT 权重不够 |

若使用 Slurm，将公共环境变量和下方**一种**训练命令放进作业脚本；资源头可使用
`#SBATCH --nodes=1`、`#SBATCH --ntasks=1`、`#SBATCH --gres=gpu:2`，分区、CPU、内存和时间按集群修改。
激活训练环境后直接运行一次 `accelerate launch`；不要启动两份相同 launcher。
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

### LingBot-VA 训练命令

使用视频与动作联合训练的 **LingBot-VA**，LeRobot policy 名为 `lingbot_va`。
训练必须用 `attn_mode=flex`；推理才使用 `torch`。下面从无 adapter 的基础 checkpoint 创建新 LoRA，
通过顶层 `--peft.*` 设置；`--policy.use_peft=true` 表示加载已有 adapter，不能单独用于创建新 adapter。
LoRA 作用于 attention 的 q/k/v/out 投影。参见
[LingBot-VA 0.6.0 接口](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/lingbot_va.mdx)
及 [PEFT 配置](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/configs/default.py)。

```bash
export MODEL_DIR="$LINGBOT_VA_MODEL_DIR"
export OUTPUT_DIR="$OUTPUT_BASE/lingbot_va_libero_safety_10k"

accelerate launch --multi_gpu --num_processes=2 --num_machines=1 \
  --mixed_precision=bf16 --dynamo_backend=no "$(command -v lerobot-train)" \
  --dataset.repo_id=local/libero_safety --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.path="$MODEL_DIR" \
  --policy.input_features="$LIBERO_INPUT_FEATURES" \
  --policy.output_features="$LIBERO_OUTPUT_FEATURES" \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.attn_mode=flex \
  --policy.wan_pretrained_path="$WAN_MODEL_DIR" --policy.text_encoder_device=cpu \
  --policy.obs_cam_keys='["observation.images.image","observation.images.image2"]' \
  --policy.camera_layout=width_concat --policy.height=128 --policy.width=128 \
  --policy.used_action_channel_ids='[0,1,2,3,4,5,6]' \
  --policy.frame_chunk_size=4 --policy.action_per_frame=4 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"IDENTITY","ACTION":"QUANTILES"}' \
  --policy.use_peft=false --policy.push_to_hub=false \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=32 \
  --peft.target_modules='["to_q","to_k","to_v","to_out.0"]' \
  --batch_size=1 --num_workers=4 --steps=10000 --seed=1000 \
  --env_eval_freq=0 --save_freq=2000 --log_freq=100 --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" --job_name=lingbot_va_libero_safety_10k
```

这里使用单臂 LIBERO 的 7 个动作通道，不适用于 RoboTwin 双臂 checkpoint。
`frame_chunk_size=4`、`action_per_frame=4` 派生出 16 步动作块及每相机 16 帧训练窗口；
`chunk_size/n_action_steps` 是派生属性，不像 π0.5 那样直接设置。
`ACTION=QUANTILES` 让训练动作和输出反归一化使用同一组新数据统计量。
LoRA 导出需要保留原始基础模型、Wan 资产和 `adapter_config.json` 中可解析的本地基础模型路径。

三者最终产物均在各自输出目录的 `checkpoints/last/pretrained_model/`；输出目录请使用新目录。
上述命令已按 0.6.0 接口核对，尚未在用户集群或其真实权重、数据上完成训练验证。

## LIBERO-Safety：socket 部署、独立测试与论文指标

训练对象是上节的 VLA；VLSA、VLPSA、OA-VLA 是下节评测的方法/安全层，分别记录结果。
当前共享 TCP 服务适配器支持 **原生 LeRobot π0.5** 和 **Panda OA-VLA**。
GR00T、LingBot-VA 的训练命令不意味着已接入本项目的安全层 socket 评测；二者仍需策略服务适配，
尤其 LingBot-VA 必须正确管理每个 episode 的时序缓存，不能直接套用 π0.5 的服务实现。
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
LeRobot 0.6.0 的 π0.5 / GR00T / LingBot-VA 训练见本页双卡离线训练一节。
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
