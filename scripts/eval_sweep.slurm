#!/usr/bin/env bash
#SBATCH -J mvsplat-codec-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=26G
#SBATCH -p batch_ugrad
#SBATCH -t 0-06:00:00
#SBATCH -x ariel-v[7]
#SBATCH -o logs/slurm/slurm-mvsplat-codec-sweep-%j.out
#SBATCH -e logs/slurm/slurm-mvsplat-codec-sweep-%j.err
#
# Scores several checkpoints of one run on the same slice of the test set, to see whether
# test quality is still improving with training.
#
#   RUN=C50k_lmbda0.0373 STEPS="4000 8000 12000 16000 20000 22000" sbatch scripts/eval_sweep.slurm
#   python scripts/summarize_sweep.py C50k_lmbda0.0373
#
# A step's checkpoint is looked up in outputs/codec/$RUN and in its resumes $RUN_r1, $RUN_r2,
# ..., so give the run's original name, not a resume folder.
#
# Only every CHUNK_EVERY-th test chunk is used (~10% of the scenes at the default 10),
# with real bitstreams and no images saved, so each checkpoint takes minutes instead of
# the half hour of a full evaluation. Compare these numbers across steps, not with
# full-test-set results. Steps already scored are skipped, so the job can be re-run later
# with more steps.

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
cd "${REPO_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-hisplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

RUN="${RUN:?set RUN to the run directory name under outputs/codec}"
STEPS="${STEPS:?set STEPS to the checkpoint steps to score}"
CODEC_CUT="${CODEC_CUT:-post_unet_skip}"
CHUNK_EVERY="${CHUNK_EVERY:-10}"
DATA_ROOT="${DATA_ROOT:-/data3/local_datasets/re10k}"

for step in ${STEPS}; do
  ckpt=$( { compgen -G "outputs/codec/${RUN}/checkpoints/*-step_${step}.ckpt" || true
            compgen -G "outputs/codec/${RUN}_r*/checkpoints/*-step_${step}.ckpt" || true; } | head -1 )
  if [[ -z "${ckpt}" ]]; then
    echo "== step ${step}: no checkpoint in outputs/codec/${RUN} or its resumes, skipped"
    continue
  fi
  name="sweep_${RUN}_s${step}"
  if [[ -f "outputs/test/${name}/scores_all_avg.json" ]]; then
    echo "== step ${step}: already scored"
    continue
  fi
  echo "== step ${step}: ${ckpt}"
  srun --ntasks=1 python -m src.main \
    +experiment=re10k_codec \
    wandb.name="${name}" \
    "dataset.roots=[${DATA_ROOT}]" \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load="${ckpt}" \
    model.encoder.codec_cut="${CODEC_CUT}" \
    model.encoder.codec_igdn_max_gain=10.0 \
    test.compute_scores=true \
    test.save_image=false \
    dataset.test_chunk_interval="${CHUNK_EVERY}" \
    "$@"
done
