#!/usr/bin/env bash
# End-to-end NEO on PhyWorld two-ball collisions.
#
#   bash scripts/phyworld_collision.sh            # full run on collision_30K
#   SMOKE=1 bash scripts/phyworld_collision.sh    # tiny simulated data, CPU, ~minutes
#
# Stages can be skipped by setting e.g. SKIP_DOWNLOAD=1 SKIP_CONVERT=1.
# Every setting below can be overridden from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

SMOKE="${SMOKE:-0}"
RAW="${RAW:-data/phyworld}"                 # PhyWorld HDF5 files
DATA="${DATA:-data/phyworld_collision}"     # converted episode files
RUNS="${RUNS:-runs/phyworld_collision}"
MODALITY="${MODALITY:-pixels}"              # pixels | states
FRAME_STEP="${FRAME_STEP:-2}"               # raw frames per program step
IMAGE_SIZE="${IMAGE_SIZE:-64}"
TRAIN_DELTAS="${TRAIN_DELTAS:-1,2,3,4}"
LENGTH_OOD_DELTAS="${LENGTH_OOD_DELTAS:-5,6,7,8}"
SEED="${SEED:-42}"

if [[ "$SMOKE" == "1" ]]; then
  RAW="${RAW_SMOKE:-data/phyworld_smoke}"; DATA="${DATA_SMOKE:-data/phyworld_collision_smoke}"
  RUNS="${RUNS_SMOKE:-runs/phyworld_collision_smoke}"
  TRAIN_EPISODES=256; TEST_EPISODES=64; VAE_EPOCHS=1; NEO_SET=(--set epochs=1 --set batch_size=16 --set num_workers=0)
  VAE_EXTRA=(--batch-size 32 --num-workers 0 --cpu); NEO_EXTRA=(--cpu); EVAL_EXTRA=(--cpu --batch-size 32)
  SKIP_DOWNLOAD=1
else
  TRAIN_EPISODES="${TRAIN_EPISODES:-60000}"; TEST_EPISODES="${TEST_EPISODES:-5000}"; VAE_EPOCHS="${VAE_EPOCHS:-30}"
  NEO_SET=(); VAE_EXTRA=(); NEO_EXTRA=(); EVAL_EXTRA=()
fi
SOURCE="${SOURCE:-$RAW/collision_30K.hdf5}"
EVAL_SOURCE="${EVAL_SOURCE:-$RAW/collision_ood_velocity.hdf5}"
STATES_FLAG=(); [[ "$MODALITY" == "states" ]] && STATES_FLAG=(--states-only)
STATE_DIM="${STATE_DIM:-$([[ "$MODALITY" == "states" ]] && echo 64 || echo 256)}"
mkdir -p "$RAW" "$DATA" "$RUNS"

# 1. Data -------------------------------------------------------------------
if [[ "$SMOKE" == "1" && ! -f "$SOURCE" ]]; then
  python -m tasks.phyworld_collision.data.simulate --preset tiny --output "$SOURCE"
  python -m tasks.phyworld_collision.data.simulate --preset ood-velocity-high --max-videos 60 --output "$EVAL_SOURCE"
fi
if [[ "${SKIP_DOWNLOAD:-0}" != "1" && ! -f "$SOURCE" ]]; then
  # needs `pip install huggingface_hub`; or download in a browser from
  # https://huggingface.co/datasets/magicr/phyworld/tree/main/id_ood_data
  hf download magicr/phyworld id_ood_data/collision_30K.hdf5 --repo-type dataset --local-dir "$RAW"
  mv "$RAW/id_ood_data/collision_30K.hdf5" "$SOURCE"
fi
if [[ ! -f "$EVAL_SOURCE" ]]; then
  # OOD speeds 4.5-6.0 (and 0-0.8 with ood-velocity-low), simulated in PhyWorld's format.
  # To use PhyWorld's own eval file instead: EVAL_SOURCE=$RAW/collision_eval.hdf5
  python -m tasks.phyworld_collision.data.simulate --preset ood-velocity-high --output "$EVAL_SOURCE"
fi
python -m tasks.phyworld_collision.data.phyworld_hdf5 "$SOURCE" --max-items 3

convert() {  # name source videos deltas episodes [extra...]
  local name=$1 source=$2 videos=$3 deltas=$4 episodes=$5; shift 5
  [[ -f "$DATA/$name.h5" ]] && { echo "exists: $DATA/$name.h5"; return; }
  python -m tasks.phyworld_collision.data.convert --source "$source" --videos "$videos" \
    --deltas "$deltas" --episodes "$episodes" --frame-step "$FRAME_STEP" --image-size "$IMAGE_SIZE" \
    ${STATES_FLAG[@]+"${STATES_FLAG[@]}"} --output "$DATA/$name.h5" "$@"
}
if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  convert train        "$SOURCE"      train "$TRAIN_DELTAS"      "$TRAIN_EPISODES" --filter id --seed 0
  convert id_test      "$SOURCE"      test  "$TRAIN_DELTAS"      "$TEST_EPISODES"  --filter id --seed 1
  convert length_ood   "$SOURCE"      test  "$LENGTH_OOD_DELTAS" "$TEST_EPISODES"  --filter id --seed 2
  convert velocity_ood "$EVAL_SOURCE" all   "$TRAIN_DELTAS"      "$TEST_EPISODES"  --filter ood-velocity --seed 3
  convert id_test_collisions "$SOURCE" test "$TRAIN_DELTAS"     "$TEST_EPISODES"  --filter id --seed 4 --collision-fraction 1.0
fi

# 2. Observation model --------------------------------------------------------
if [[ -z "${OBS_CKPT:-}" ]]; then
  python -m tasks.phyworld_collision.observation_pretraining --modality "$MODALITY" --state-dim "$STATE_DIM" \
    --train-h5 "$DATA/train.h5" --test-h5 "$DATA/id_test.h5" --output-root "$RUNS/observation" \
    --epochs "$VAE_EPOCHS" --seed "$SEED" ${VAE_EXTRA[@]+"${VAE_EXTRA[@]}"}
  OBS_CKPT="$(ls -td "$RUNS"/observation/"$MODALITY"-*/ | head -1)checkpoints/best.pt"
fi
echo "observation checkpoint: $OBS_CKPT"

# 3. NEO ----------------------------------------------------------------------
if [[ -z "${NEO_CKPT:-}" ]]; then
  python -m tasks.phyworld_collision.theorizer_runner --train-h5 "$DATA/train.h5" --test-h5 "$DATA/id_test.h5" \
    --observation-checkpoint "$OBS_CKPT" --output-root "$RUNS/neo" --seed "$SEED" ${NEO_SET[@]+"${NEO_SET[@]}"} ${NEO_EXTRA[@]+"${NEO_EXTRA[@]}"}
  NEO_CKPT="$(ls -td "$RUNS"/neo/*/*/ | head -1)checkpoints/best.pt"
fi
echo "NEO checkpoint: $NEO_CKPT"

# 4. Evaluation ---------------------------------------------------------------
python -m tasks.phyworld_collision.evaluation --checkpoint "$NEO_CKPT" --output-dir "$RUNS/evaluation-seed$SEED" \
  --split id="$DATA/id_test.h5" --split id_collisions="$DATA/id_test_collisions.h5" \
  --split length_ood="$DATA/length_ood.h5" --split velocity_ood="$DATA/velocity_ood.h5" ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}
