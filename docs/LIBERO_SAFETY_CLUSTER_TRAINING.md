# LIBERO-Safety 数据准备与集群 VLA 训练

本文提供两条完整但格式不同的路径：

1. 官方两视角示范 → 原生 LeRobot π₀.₅ 训练 → 本地 LeRobot baseline 推理。
2. baseline 在仿真采集四视角、配对 OBB → 本项目 OA-VLA 训练 → 与真机相同的 TCP + Voxblox + MCBF 推理。

LeRobot 的 `policy.type=pi05` 只训练原生动作模型，不会自动训练本项目的 SAM 3 配对 OBB 分支。
当前 OA-VLA 没有注册成 LeRobot `PreTrainedPolicy` 插件；第二条使用已有 `oa-vla-train`/DDP，
不能将它描述成 LeRobot 原生联合训练。两种权重不能相互替换。

本地推理命令及四视角定义见 [本地实验](LIBERO_SAFETY_LOCAL.md)。以下所有命令均从本项目根目录运行。
示例假设 Linux + CUDA + Slurm，先替换登录地址、共享目录、账户/分区和 GPU 数量。

## 1. 已核实的数据与版本

官方训练数据：[LIBERO-Safety/libero_safety](https://huggingface.co/datasets/LIBERO-Safety/libero_safety)。
截至 2026-11-03，实际 `meta/info.json` 声明：

| 字段 | 内容 |
| --- | --- |
| `codebase_version` | `v2.1` |
| 规模 | 19,664 episodes；3,443,735 frames；15 种语言任务 |
| 帧率 | 20 Hz |
| `observation.image` | agentview，256×256 RGB，AV1 视频 |
| `observation.wrist_image` | wrist，256×256 RGB，AV1 视频 |
| `observation.state` | 8 维，EEF xyz + axis-angle + 两个夹爪关节位置 |
| `actions` | 7 维；前 6 维为米/弧度增量，最后一维为夹爪命令 |
| 划分 | 元数据仅提供 train；不能将该划分当作独立测试集 |

动作单位经过公开 `episode_000000.parquet` 抽样核对：相邻 EEF xyz 差与动作 xyz 基本相等，
各轴拟合增益约 1.0；不能把这些米制位移原样传给默认归一化 OSC 控制器。
本项目转换工具按默认 OSC 的 xyz 0.05 m、旋转 0.5 rad 缩放，将前 6 维转换为归一化命令，
保留夹爪命令，且同步变换统计。该抽样不是对全部示范动力学的复现验证；正式训练前仍应做轨迹回放检查。

官方未提供四路图像、双视角配对 OBB 或本项目的体素地图。训练数据约 19.1 GB；
仿真资产另约 10.7 GB，训练离线 BC 不需要仿真资产。
论文用 L0/L1 训练，L2 作 OOD 测试；不得把本地新增的 L2 采集混入训练。

本文固定 **LeRobot v0.4.4**。这是已核查接口的版本，不混用后续 main 分支的并行训练参数。
该版本使用 `accelerate launch`；π₀.₅ 的 `[pi]` extra 使用专门的 Transformers 分支，
与本项目 OA-VLA 固定的 Transformers 4.53.2 应分环境安装。

## 2. 在有网络的机器准备环境和文件

```bash
conda create -n lerobot-safety-train python=3.11 -y
conda activate lerobot-safety-train
mkdir -p outputs/deps outputs/downloads outputs/weights
git clone --branch v0.4.4 --depth 1 https://github.com/huggingface/lerobot.git outputs/deps/lerobot
# 普通安装便于后面 conda-pack；不要用 -e 把代码留在环境外。
python -m pip install './outputs/deps/lerobot[pi]' 'huggingface_hub[cli]<1' pyarrow av conda-pack
git -C outputs/deps/lerobot rev-parse HEAD > outputs/lerobot_commit.txt
python -m pip freeze > outputs/lerobot_environment.txt
```

按照集群 CUDA/驱动选择 PyTorch 包。先在登录节点确认 `torch.cuda.is_available()`、
`lerobot-train --help` 和 AV1 解码可用；不在训练任务内临时联网升级依赖。

先在 Hugging Face 接受 [PaliGemma tokenizer 的访问条款](https://huggingface.co/google/paligemma-3b-pt-224)，
然后登录。tokenizer 不可访问时，训练前处理就会失败。

```bash
export HF_HOME="$PWD/outputs/hf_cache"
hf auth login
hf download lerobot/pi05_base --local-dir outputs/weights/lerobot_pi05_base
hf download google/paligemma-3b-pt-224 \
  --include 'tokenizer*' 'special_tokens_map.json' 'config.json'
```

`lerobot/pi05_base` 是本路径需要的原生 PyTorch checkpoint；
官方 `LIBERO-Safety/pi05_libero_safety` 的 OpenPI/JAX 文件不是同一格式。
下载结束应能看到 base 的 `config.json`、模型权重以及 pre/post processor 配置和状态文件。
基座模型默认是 32 维、三相机配置，后面的训练命令会显式改成两相机、8 维状态、7 维动作。

下载完整数据并记录不可变 revision：

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'LIBERO-Safety/libero_safety'
revision = HfApi().dataset_info(repo).sha
Path('outputs/libero_safety_data_revision.txt').write_text(revision + '\n')
snapshot_download(repo, repo_type='dataset', revision=revision,
                  local_dir='outputs/downloads/libero_safety_raw')
PY
```

不要只下载 `data/*.parquet`：还需要 `meta/` 和所有 `videos/`。

## 3. 规范化字段、动作单位和数据版本

原始副本保留不动；转换产生派生副本。建议数据盘预留至少 100 GB，迁移视频时有额外临时文件。

```bash
python scripts/normalize_libero_safety_dataset.py \
  --source outputs/downloads/libero_safety_raw \
  --output outputs/datasets/local/libero_safety \
  --action-units metric-delta --link-videos
```

该工具完成：

- `actions` → `action`，并对 xyz / 0.05、旋转 / 0.5；最后一维不变。
- 两路图像 → `observation.images.image`、`observation.images.image2`。
- 同步改 parquet schema、JSON/JSONL 特征和统计，以及视频目录名称。
- 检查 episode 与视频数量；失败时保留 `INCOMPLETE`，该副本不得用于训练。

`--link-videos` 只对不可变视频建立硬链接，跨文件系统则复制。源数据被保留。
如果自己采集的数据本来就是归一化 OSC 命令，应使用 `--action-units normalized-osc`；禁止再次缩放。
改变了 controller 输出尺度时也必须重新适配，不能继续使用本文默认除数。

不能只用训练器的 `--rename_map` 重命名 `actions`：LeRobot 在重命名前就按 `action` 构造 chunk 时间索引，
这样可能无法得到正确的动作块。

LeRobot v0.4.4 的训练数据格式为 v3.0，继续用其官方迁移器：

```bash
cp outputs/datasets/local/libero_safety/normalization.json outputs/libero_safety_normalization.json
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id local/libero_safety --root "$PWD/outputs/datasets" \
  --push-to-hub false
cp outputs/libero_safety_normalization.json outputs/datasets/local/libero_safety/normalization.json
python scripts/compute_lerobot_quantiles.py --dataset outputs/datasets/local/libero_safety
```

注意迁移器的 `--root` 是 repo ID 的**父目录**：它会拼接 `local/libero_safety`。
迁移器在派生数据旁保留 `_old` 副本；不要反复对同一路径运行它。
官方 v2.1 统计只有 mean/std/min/max，缺少 native π₀.₅ 默认需要的 q01/q99；
最后一个工具从迁移后的训练向量精确计算分位数并写入 `meta/stats.json`，不读取视频。
不要在混有验证/测试 episode 的目录上重算训练统计。

在 GPU 训练前先验证真实解码及动作 chunk：

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/libero_safety', root='outputs/datasets/local/libero_safety',
    delta_timestamps={'action': [i / 20 for i in range(50)]}, video_backend='pyav')
x = ds[0]
print({k: getattr(v, 'shape', None) for k, v in x.items()})
assert tuple(x['action'].shape) == (50, 7)
assert tuple(x['observation.state'].shape) == (8,)
assert tuple(x['observation.images.image'].shape) == (3, 256, 256)
assert tuple(x['observation.images.image2'].shape) == (3, 256, 256)
print(x['task'])
PY
```

同时抽查视频朝向、颜色、state 与动作对齐。本地扩展使用左上原点 RGB；
如果后续不同 revision 改了图像约定，应在数据与推理两端统一后再训练。

## 4. 上传集群

以下 `researcher@login.cluster` 和 `/shared/researcher` 都是示例，必须替换。

```bash
rsync -a --info=progress2 --exclude='.git' --exclude='outputs' --exclude='.venv' \
  ./ researcher@login.cluster:/shared/researcher/safety/
rsync -a --info=progress2 outputs/datasets/local/libero_safety/ \
  researcher@login.cluster:/shared/researcher/data/libero_safety/
rsync -a --info=progress2 outputs/weights/lerobot_pi05_base/ \
  researcher@login.cluster:/shared/researcher/weights/lerobot_pi05_base/
rsync -a --info=progress2 outputs/hf_cache/hub/ \
  researcher@login.cluster:/shared/researcher/hf_cache/hub/
rsync -a outputs/lerobot_commit.txt outputs/lerobot_environment.txt \
  outputs/libero_safety_data_revision.txt outputs/libero_safety_normalization.json \
  researcher@login.cluster:/shared/researcher/safety/
```

只同步 `HF_HOME/hub/` 缓存，不需要上传本机登录 token。
共享文件系统不适合大量视频读取时，再把数据复制到作业节点本地 SSD，并相应修改 `SAFETY_DATASET`。

计算节点不能联网时，可以在兼容的 Linux 机器打包刚才的普通安装环境：

```bash
conda pack -n lerobot-safety-train -o outputs/lerobot-safety-train.tar.gz
rsync -a --info=progress2 outputs/lerobot-safety-train.tar.gz \
  researcher@login.cluster:/shared/researcher/
# 在集群登录节点：
mkdir -p /shared/researcher/envs/lerobot-safety-train
tar -xzf /shared/researcher/lerobot-safety-train.tar.gz \
  -C /shared/researcher/envs/lerobot-safety-train
source /shared/researcher/envs/lerobot-safety-train/bin/activate
conda-unpack
```

环境包需匹配 CPU 架构、Linux 系统库和集群 NVIDIA 驱动。
如果登录节点可以联网，也可按第 2 节直接重建相同环境，保留相同版本记录。

## 5. 原生 LeRobot 单卡试跑与 Slurm 多卡训练

先做一个 2-step 单卡作业，确认权重、tokenizer、视频解码和保存均正常。
以下示例使用 80 GB GPU 作为起点；实际显存随训练范围、batch 和软件版本变化，不能只按 GPU 数推断。

```bash
export HF_HOME=/shared/researcher/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
lerobot-train \
  --dataset.repo_id=local/libero_safety \
  --dataset.root=/shared/researcher/data/libero_safety \
  --dataset.video_backend=pyav \
  --policy.path=/shared/researcher/weights/lerobot_pi05_base \
  --policy.input_features='{"observation.images.image":{"type":"VISUAL","shape":[3,256,256]},"observation.images.image2":{"type":"VISUAL","shape":[3,256,256]},"observation.state":{"type":"STATE","shape":[8]}}' \
  --policy.output_features='{"action":{"type":"ACTION","shape":[7]}}' \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true --policy.train_expert_only=true \
  --policy.n_action_steps=1 --policy.push_to_hub=false \
  --batch_size=1 --num_workers=0 --steps=2 --save_freq=2 \
  --log_freq=1 --eval_freq=0 --wandb.enable=false \
  --output_dir=outputs/train/libero_safety_smoke_01
```

`train_expert_only=true` 冻结 VLM，只微调 action expert 等相应参数；这是资源较省的起始配置，
不是论文全量微调的等价复现。全量微调需设为 false 并重新测量显存。
`n_action_steps=1` 与本地“每步重新观测”的执行协议一致；训练 chunk 仍为 50。
`eval_freq=0` 避免误调用 LeRobot 原版 LIBERO 评估器，安全基准评估用本项目独立入口。

多卡单节点模板为 [train_libero_safety_pi05.slurm](../scripts/cluster/train_libero_safety_pi05.slurm)。
它已包含输入/输出特征覆盖、离线模式和数据预检查。

```bash
cd /shared/researcher/safety
mkdir -p logs
export SAFETY_PROJECT=/shared/researcher/safety
export SAFETY_DATASET=/shared/researcher/data/libero_safety
export SAFETY_BASE_MODEL=/shared/researcher/weights/lerobot_pi05_base
export SAFETY_HF_CACHE=/shared/researcher/hf_cache
sbatch scripts/cluster/train_libero_safety_pi05.slurm
```

模板 `--gres=gpu:4` 与 `--num_processes=4` 必须一起修改。4 个 DDP 进程各持一份模型，
不会自动将单模型显存分摊到 4 张卡；该模板也不是多节点/FSDP。
batch_size=1 表示每进程 1 个样本，4 卡全局 batch 为 4。
Slurm 的 `logs/` 必须在提交前存在，且作业继承已激活环境的 PATH。

检查 `squeue -u "$USER"`、日志和 `outputs/train/.../checkpoints/`。
保存目录通常为 `checkpoints/005000/pretrained_model` 等，保留 processors 和训练状态，
不能只拷贝 `model.safetensors`。

恢复训练使用保存的完整配置和 training_state；不要再次传 base model 的 `--policy.path`：

```bash
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 \
  "$(command -v lerobot-train)" \
  --config_path=outputs/train/你的作业/checkpoints/005000/pretrained_model/train_config.json \
  --resume=true
```

同样保留 GPU 数、batch 和原始数据路径。首次恢复前确认 checkpoint 下存在对应训练状态。

## 6. 下载训练结果并本地推理

```bash
rsync -a --info=progress2 \
  researcher@login.cluster:/shared/researcher/safety/outputs/train/你的作业/checkpoints/030000/pretrained_model/ \
  outputs/models/pi05_safety/pretrained_model/
```

本地安装同版本 native LeRobot 环境和仿真依赖，然后：

```bash
python scripts/evaluate_oa_vla_libero_safety.py \
  --libero-root outputs/deps/LIBERO-Safety --backend lerobot --mode baseline \
  --policy outputs/models/pi05_safety/pretrained_model --device cuda \
  --suite obstacle_avoidance --level 0 --task 0 --episodes 0 1 2 \
  --output outputs/libero_safety/native_pi05_run1
```

原生 LeRobot 环境保留其 `[pi]` Transformers 版本；仿真端只需将 `oa-vla` 以 `--no-deps` 安装，
用于轻量公共工具，不在这里加载自定义 OA-VLA 网络。

## 7. 训练当前四视角 OA-VLA 并接回完整安全层

先按 [本地采集说明](LIBERO_SAFETY_LOCAL.md#lerobot-baseline-与四视角训练采集)
生成 `outputs/datasets/panda_oavla_train/*.shard`。该步骤只接收完整安全成功的示范；
baseline 不成功时没有动作训练数据，需先改善策略或另行提供专家采集，不能把失败 rollout 冒充专家示范。
分割真值只用于训练标签，不进入评估模型输入。

将 `.shard` 目录整体上传；同时准备现有**同结构 OA-VLA**权重、SentencePiece `tokenizer.model`，
需要时再准备 SAM 3 decoder 初始化。单独创建 OA-VLA 训练环境：

```bash
conda create -n oavla-panda-train python=3.11 -y
conda activate oavla-panda-train
python -m pip install -e './oa-vla[data,serve]'
python -m oa_vla.train --dataset /shared/researcher/data/panda_oavla_train \
  --stage joint --output outputs/train/oavla_panda_validate --validate-only
```

训练器从数据自动推导 state_dim=8、logical_action_dim=7、horizon=8；内部仍 padding 到 32。
它会保留 `.shard` 的相机映射、状态/动作语义及周期，避免导出后丢失推理契约。

```bash
torchrun --standalone --nproc_per_node=4 -m oa_vla.train --distributed \
  --dataset /shared/researcher/data/panda_oavla_train \
  --output outputs/train/oavla_panda_joint \
  --pretrained /shared/researcher/weights/oavla_existing/model.safetensors \
  --tokenizer-model /shared/researcher/weights/tokenizer.model \
  --stage joint --device cuda --precision bfloat16 \
  --batch-size 1 --num-workers 2 --epochs 20 --save-every 1
```

上述 `oavla_existing` 可以来自同结构的现有模型，用于参数初始化；会重新计算 Panda 数据的归一化统计，
并重新学习不同 embodiment 的动作与检测。它不等于直接将双臂权重作为单臂策略执行。
`--pretrained` 严格检查参数名和形状；原生 LeRobot 或 JAX 权重不能直接放在该位置。
若现有模型架构配置不同，先核对 config，不用 `--allow-random-init` 绕过正式初始化。

可通过 `--stage action` / `--stage detection` 分阶段训练；joint/detection 必须有真实审核或明确仿真来源的配对 OBB 标签。
训练输出包含 `config.json`、`metadata.json`、`processor.json`、tokenizer、模型和 training_state。
恢复使用 `--resume /原输出目录`；不要同时提供 `--pretrained`。

下载整个 OA-VLA 导出目录后启动 `oa-vla-serve`，运行本地 `--backend oa-vla --mode safety`。
这才是本项目当前“联合模型 + 感知 + MCBF/QP”的完整评估路径。

## 8. 复现记录

每次保存：数据 revision、规范化动作除数、v2.1→v3 迁移版本、相机/状态约定、训练 episode 列表、
LeRobot commit、环境清单、随机种子、GPU/批大小、训练阶段、完整 checkpoint 和评估初始化索引。
正式评估使用固定时限、相同初始状态和完整 episode，报告安全成功、违规、超时、hold、干预与时延。
本地新增四视角/depth/安全层属于额外输入和执行协议，需与论文原始协议区分。

接口依据：[LeRobot v0.4.4 配置](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/pi05/configuration_pi05.py)、
[训练入口](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/scripts/lerobot_train.py)、
[官方 v3 迁移器](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py)、
[官方数据元信息](https://huggingface.co/datasets/LIBERO-Safety/libero_safety/blob/main/meta/info.json)。
