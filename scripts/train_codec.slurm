#!/usr/bin/env bash
#SBATCH -J mvsplat-codec-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=26G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -x ariel-v[7]
#SBATCH -o logs/slurm/slurm-mvsplat-codec-train-%j.out
#SBATCH -e logs/slurm/slurm-mvsplat-codec-train-%j.err
#
# Fine-tunes the released MVSplat re10k checkpoint with the feature codec enabled.
#
#   sbatch scripts/train_codec.slurm                        # cut A: pre_unet (default)
#   CODEC_CUT=post_unet sbatch scripts/train_codec.slurm    # cut B: after the refine U-Net
#   CODEC_CUT=post_unet_skip sbatch scripts/train_codec.slurm   # cut C: after it, skip inputs sent too
#   LMBDA=0.1493 sbatch scripts/train_codec.slurm
#   BATCH_PER_GPU=1 sbatch scripts/train_codec.slurm        # on OOM; accum adjusts
#   MAX_STEPS=50 CKPT_EVERY=50 sbatch scripts/train_codec.slurm   # smoke test
#   RESUME=true CKPT=outputs/codec/<run>/checkpoints/<x>.ckpt sbatch scripts/train_codec.slurm
#       # continue an interrupted run: restores optimizer state and step; pass the same
#       # CODEC_CUT / LMBDA / TOTAL_SAMPLES so max_steps and warm_up come out identical
#
# Default is 1 GPU (accum 32 at BATCH_PER_GPU=2) on whichever normal-GPU node of
# batch_ugrad (v6-v12) has room, minus the ones -x drops. (-w with several nodes would
# demand all of them at once.) To use another GPU count or a specific node, override the
# #SBATCH lines on the command line (they take precedence):
#   sbatch --gres=gpu:normal:2 --ntasks-per-node=2 -w ariel-v8 scripts/train_codec.slurm
# accum then drops so the effective batch, max_steps and warm_up stay the same.
# Finishing max_steps within -t needs max_steps / 144 steps an hour for 6 days (~350 for
# 50k); if a run falls short, resume it from its last checkpoint.
#
# --ntasks-per-node must equal --gres=gpu:N; srun then starts one DDP rank per GPU.
# accumulate_grad_batches, max_steps and warm_up_steps are all derived below, so
# changing the GPU count only changes wall clock.
# Anything after the script name is forwarded to hydra as extra overrides.

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
mkdir -p logs/slurm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-hisplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Lightning writes each checkpoint to a temp file first (tempfile.mkstemp), which lands on
# the node's /tmp unless TMPDIR says otherwise. A full /tmp on ariel-v11 killed a run at
# its only save. Keeping it on ceph also makes the final move a plain rename.
export TMPDIR="${REPO_DIR}/tmp"
mkdir -p "${TMPDIR}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
LMBDA="${LMBDA:-0.0373}"
CODEC_CUT="${CODEC_CUT:-pre_unet}"   # pre_unet | post_unet | post_unet_skip
CKPT="${CKPT:-checkpoints/re10k.ckpt}"
RESUME="${RESUME:-false}"            # true: CKPT is a checkpoint of this very run, not just weights
# main.yaml's default interval (20000) is above max_steps for reduced-budget runs, which
# would end without ever writing a checkpoint. Checked against MAX_STEPS below.
CKPT_EVERY="${CKPT_EVERY:-2000}"
# Pin the output directory to this job. Hydra's default is outputs/<date>/<HH-MM-SS>,
# so two jobs submitted in the same second share a directory and overwrite each
# other's checkpoints, which have identical filenames.
RUN_NAME="${RUN_NAME:-${CODEC_CUT}_lmbda${LMBDA}_job${SLURM_JOB_ID:-$$}}"
OUTPUT_DIR="outputs/codec/${RUN_NAME}"
# Must hold train/ and test/, each with *.torch chunks and an index.json.
DATA_ROOT="${DATA_ROOT:-/data3/local_datasets/re10k}"

# Derive accumulation from the target effective batch, so the GPU count only affects
# wall clock. 64 is what the reference codec stage used (4 GPUs x 4 x accum 4), and
# lr 2e-4 was tuned there.
TARGET_BATCH="${TARGET_BATCH:-64}"
ACCUM="${ACCUM:-$(( TARGET_BATCH / (NUM_GPUS * BATCH_PER_GPU) ))}"

# Fix the sample budget instead of the step count, so changing the GPU count does not
# change how much the model actually sees. 4.2M = 14 x 300k, the baseline's budget.
TOTAL_SAMPLES="${TOTAL_SAMPLES:-4200000}"
EFFECTIVE_BATCH=$(( NUM_GPUS * BATCH_PER_GPU * ACCUM ))
# Override with MAX_STEPS=..., not a trailing trainer.max_steps=... : hydra would then
# see the key twice and the echo below would report the wrong number.
MAX_STEPS="${MAX_STEPS:-$(( TOTAL_SAMPLES / EFFECTIVE_BATCH ))}"

# The view sampler widens the context-view gap over warm_up_steps. Its default of
# 150k assumes the baseline's 300k-step run; hold it at the same half-way point of
# whatever this run's length is, or the far-apart views are never seen.
WARM_UP="${WARM_UP:-$(( MAX_STEPS / 2 ))}"

if (( CKPT_EVERY > MAX_STEPS )); then
  echo "CKPT_EVERY=${CKPT_EVERY} exceeds MAX_STEPS=${MAX_STEPS}: the run would finish" \
       "without saving a checkpoint. Set CKPT_EVERY<=${MAX_STEPS}." >&2
  exit 1
fi
if (( ACCUM < 1 )); then
  echo "TARGET_BATCH=${TARGET_BATCH} is below ${NUM_GPUS} GPUs x ${BATCH_PER_GPU};" \
       "set ACCUM explicitly" >&2
  exit 1
fi
if (( ${SLURM_NTASKS_PER_NODE:-NUM_GPUS} != NUM_GPUS )); then
  echo "--ntasks-per-node (${SLURM_NTASKS_PER_NODE:-?}) must equal the GPU count" \
       "(${NUM_GPUS}); fix both #SBATCH lines" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/train" ]]; then
  echo "dataset not found: ${DATA_ROOT}/train" >&2
  exit 1
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output dir already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

echo "gpus=${NUM_GPUS} batch/gpu=${BATCH_PER_GPU} accum=${ACCUM}" \
     "-> effective batch ${EFFECTIVE_BATCH}, max_steps=${MAX_STEPS}," \
     "warm_up=${WARM_UP}, lmbda=${LMBDA}, codec_cut=${CODEC_CUT}," \
     "ckpt_every=${CKPT_EVERY}, resume=${RESUME}, out=${OUTPUT_DIR}"

exec srun python -m src.main \
  +experiment=re10k_codec \
  wandb.name="re10k_codec_${CODEC_CUT}_lmbda${LMBDA}" \
  output_dir="${OUTPUT_DIR}" \
  "dataset.roots=[${DATA_ROOT}]" \
  checkpointing.load="${CKPT}" \
  checkpointing.resume="${RESUME}" \
  checkpointing.every_n_train_steps="${CKPT_EVERY}" \
  model.encoder.codec_cut="${CODEC_CUT}" \
  data_loader.train.batch_size="${BATCH_PER_GPU}" \
  trainer.accumulate_grad_batches="${ACCUM}" \
  trainer.max_steps="${MAX_STEPS}" \
  dataset.view_sampler.warm_up_steps="${WARM_UP}" \
  train.lmbda="${LMBDA}" \
  "$@"
