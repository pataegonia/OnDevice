#!/usr/bin/env bash
#SBATCH -J mvsplat-codec-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ugrad
#SBATCH -t 0-12:00:00
#SBATCH -x ariel-v[7]
#SBATCH -o logs/slurm/slurm-mvsplat-codec-eval-%j.out
#SBATCH -e logs/slurm/slurm-mvsplat-codec-eval-%j.err
#
#   CKPT=outputs/.../checkpoints/last.ckpt RUN_NAME=lmbda0.597 sbatch scripts/eval_codec.slurm
#   CODEC_CUT=post_unet CKPT=outputs/codec/post_unet_.../last.ckpt sbatch scripts/eval_codec.slurm
#   EXPERIMENT=re10k CKPT=checkpoints/re10k.ckpt sbatch scripts/eval_codec.slurm
#
# CODEC_CUT must match the cut the checkpoint was trained with (it is in the output
# dir name); a mismatch re-initialises to_gaussians.0 instead of failing.
#
# SINGLE GPU ONLY. ModelWrapper.on_test_end has no rank-zero guard and the test path
# has no DistributedSampler, so multiple ranks race on the score JSON and average over
# duplicated batches. To sweep lambdas, submit one job per lambda.

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"
mkdir -p logs/slurm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-hisplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EXPERIMENT="${EXPERIMENT:-re10k_codec}"
CODEC_CUT="${CODEC_CUT:-pre_unet}"   # pre_unet | post_unet | post_unet_skip
CKPT="${CKPT:?set CKPT to the checkpoint to evaluate}"
# Must hold train/ and test/, each with *.torch chunks and an index.json.
DATA_ROOT="${DATA_ROOT:-/data3/local_datasets/re10k}"
# Scores land in test.output_path/<wandb.name>, so without a distinct name every
# evaluation overwrites the previous one's scores_all_avg.json. Pass RUN_NAME to
# label the run (e.g. RUN_NAME=lmbda2.3882).
RUN_NAME="${RUN_NAME:-${EXPERIMENT}_job${SLURM_JOB_ID:-$$}}"

if [[ ! -f "${CKPT}" ]]; then
  echo "checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/test" ]]; then
  echo "dataset not found: ${DATA_ROOT}/test" >&2
  exit 1
fi

echo "experiment=${EXPERIMENT} codec_cut=${CODEC_CUT} run=${RUN_NAME} ckpt=${CKPT} data=${DATA_ROOT}"

exec srun --ntasks=1 python -m src.main \
  "+experiment=${EXPERIMENT}" \
  wandb.name="${RUN_NAME}" \
  "dataset.roots=[${DATA_ROOT}]" \
  mode=test \
  dataset/view_sampler=evaluation \
  checkpointing.load="${CKPT}" \
  model.encoder.codec_cut="${CODEC_CUT}" \
  test.compute_scores=true \
  "$@"
