#!/usr/bin/env bash
#SBATCH --job-name=libero-safety-pi05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/libero-safety-%j.out
set -euo pipefail

# Activate the pinned LeRobot v0.4.4 environment BEFORE sbatch, or add the site's
# module/conda activation here. Adapt partition/account/GPU options to your site.
: "${SAFETY_PROJECT:?Set SAFETY_PROJECT to the uploaded project directory}"
: "${SAFETY_DATASET:?Set SAFETY_DATASET to the migrated canonical v3 dataset}"
: "${SAFETY_BASE_MODEL:?Set SAFETY_BASE_MODEL to the native lerobot/pi05_base snapshot}"
: "${SAFETY_HF_CACHE:?Set SAFETY_HF_CACHE to a prepopulated HF_HOME incl. tokenizer}"
export HF_HOME="$SAFETY_HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$SAFETY_PROJECT"

python - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ['SAFETY_DATASET'])
info=json.loads((root/'meta/info.json').read_text())
stats=json.loads((root/'meta/stats.json').read_text())
assert info['codebase_version']=='v3.0'
assert info['features']['action']['shape']==[7]
assert info['features']['observation.state']['shape']==[8]
for key in ('action','observation.state'):
    assert all(q in stats[key] for q in ('q01','q99')), 'Run compute_lerobot_quantiles.py'
assert not (root/'INCOMPLETE').exists()
PY

accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 \
  "$(command -v lerobot-train)" \
  --dataset.repo_id=local/libero_safety \
  --dataset.root="$SAFETY_DATASET" \
  --dataset.video_backend=pyav \
  --policy.path="$SAFETY_BASE_MODEL" \
  --policy.input_features='{"observation.images.image":{"type":"VISUAL","shape":[3,256,256]},"observation.images.image2":{"type":"VISUAL","shape":[3,256,256]},"observation.state":{"type":"STATE","shape":[8]}}' \
  --policy.output_features='{"action":{"type":"ACTION","shape":[7]}}' \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.n_action_steps=1 \
  --policy.push_to_hub=false \
  --batch_size=1 --num_workers=4 --steps=30000 \
  --eval_freq=0 --save_freq=5000 --log_freq=100 \
  --wandb.enable=false \
  --output_dir="outputs/train/libero_safety_pi05_${SLURM_JOB_ID:-local}"
