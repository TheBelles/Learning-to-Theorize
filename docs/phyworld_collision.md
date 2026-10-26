# NEO on PhyWorld two-ball collisions

This task trains the Neural Theorizer (NEO) on the collision videos of
PhyWorld ([Kang et al., 2024](https://github.com/phyworld/phyworld)). It lives
in `src/tasks/phyworld_collision/` and does not change the paper's tasks or
scripts.

## The task

| Concept | Choice |
| --- | --- |
| Observation | Two frames, `t - k` and `t` (`k = --frame-step`, default 2), so it shows positions **and** velocities. Pixels: `64×64×6` uint8. States: 10 numbers (x, y, vx, vy, r per ball). |
| Pair | `x = obs(t)`, `y = obs(t + delta·k)` |
| Episode | Support pair from video A, query pair from a different video B, **same delta**. The only thing they share is "advance time by delta steps", i.e. the physics. |
| Program | NEO picks up to `max_transition_length` codes from a small codebook. Ideally it learns one "step" primitive and uses `delta` of them. |
| Selection / test | Program (and its length) inferred on the support pair, executed on the query pair. |

Generalisation tests, all from videos never seen in training:

* **ID**: same speeds/radii as training (`id_test.h5`), plus a version where
  every query window contains the contact (`id_test_collisions.h5`).
* **Length OOD**: `delta` 5–8 while training used 1–4 (`length_ood.h5`).
  This needs composition of learned steps.
* **Velocity OOD**: speeds outside [1, 4] (`velocity_ood.h5`), the setting in
  which video models fail in the PhyWorld paper.

## Install

Use the repository environment (`environments/README.md`), then
`pip install imageio imageio-ffmpeg huggingface_hub` (the last two only for
`--render video` and for downloading).

Check everything works on CPU before using real data (1–3 minutes):

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_phyworld_collision.py
SMOKE=1 bash scripts/phyworld_collision.sh      # same pipeline as below on tiny simulated data
```

## Full run

```bash
bash scripts/phyworld_collision.sh
```

does all of the steps below; set `MODALITY=states` for the fast state-vector
version (recommended first). Each step on its own:

**1. Data.** Download `collision_30K.hdf5` from
[magicr/phyworld](https://huggingface.co/datasets/magicr/phyworld/tree/main/id_ood_data)
and look at its layout:

```bash
python -m tasks.phyworld_collision.data.phyworld_hdf5 data/phyworld/collision_30K.hdf5
```

OOD sets in the same format can be simulated (closed-form 1D collision,
PhyWorld's placement and parameter grids, Box2D's inelastic threshold):

```bash
python -m tasks.phyworld_collision.data.simulate --preset ood-velocity-high --output data/phyworld/collision_ood_velocity.hdf5
# presets: id-30k id-eval ood-velocity-low ood-velocity-high ood-radius-low ood-radius-high tiny
```

PhyWorld's own `collision_eval.hdf5` works too: `--filter ood-velocity`
keeps the videos whose speeds are outside [1, 4].

**2. Episodes.** One file per split; `--videos train/test` splits by video.

```bash
SRC=data/phyworld/collision_30K.hdf5
C="python -m tasks.phyworld_collision.data.convert --frame-step 2 --image-size 64"
$C --source $SRC --videos train --filter id --deltas 1,2,3,4 --episodes 60000 --output data/phyworld_collision/train.h5
$C --source $SRC --videos test  --filter id --deltas 1,2,3,4 --episodes 5000  --output data/phyworld_collision/id_test.h5
$C --source $SRC --videos test  --filter id --deltas 5,6,7,8 --episodes 5000  --output data/phyworld_collision/length_ood.h5
$C --source data/phyworld/collision_ood_velocity.hdf5 --videos all --filter ood-velocity \
   --deltas 1,2,3,4 --episodes 5000 --output data/phyworld_collision/velocity_ood.h5
```

Frames are drawn from the stored positions by default (`--render positions`,
PhyWorld's own drawing code at 64 px). `--render video` decodes the MP4s and
area-downsamples them instead. `--states-only` skips pixels.
`--collision-fraction 0.5` makes half the windows contain the contact
(naturally about a third do).

**3. Observation model** (frozen during NEO training):

```bash
python -m tasks.phyworld_collision.observation_pretraining \
  --train-h5 data/phyworld_collision/train.h5 --test-h5 data/phyworld_collision/id_test.h5 \
  --output-root runs/phyworld_collision/observation --epochs 30
```

Open `reconstructions_epoch*.png` (left: input, right: reconstruction; each
cell shows frame `t-k` and frame `t`). Both balls must be sharp before going
on. The loss weights ball pixels 10× (`--foreground-weight`) so the model
cannot get away with drawing an empty white image.

**4. NEO:**

```bash
python -m tasks.phyworld_collision.theorizer_runner \
  --train-h5 data/phyworld_collision/train.h5 --test-h5 data/phyworld_collision/id_test.h5 \
  --observation-checkpoint runs/phyworld_collision/observation/pixels-*/checkpoints/best.pt \
  --output-root runs/phyworld_collision/neo --seed 42
```

Override any field of `experiment_config.TrainingConfig` with
`--set name=value`. `checkpoints/best.pt` has the lowest ID query loss.

**5. Evaluation:**

```bash
python -m tasks.phyworld_collision.evaluation --checkpoint runs/.../checkpoints/best.pt \
  --output-dir runs/phyworld_collision/eval \
  --split id=data/phyworld_collision/id_test.h5 \
  --split length_ood=data/phyworld_collision/length_ood.h5 \
  --split velocity_ood=data/phyworld_collision/velocity_ood.h5
```

## Reading the results

`evaluation.json` and the printed table give, per split and separately for
query windows with and without the contact:

| Metric | Meaning |
| --- | --- |
| `query_loss` | Error of the transferred program on the query pair (main L2T metric). |
| `length_matches_delta` | Share of programs whose length equals `delta`. High = NEO uses one primitive per time step. |
| `position_error`, `velocity_error` | Ball position at the last frame and velocity, world units (world = 10 wide), from tracking the red/blue balls in the predicted frames. |
| `constant_velocity_*`, `copy_*` | Same errors for two references: extrapolating the true input motion with no collision, and predicting no motion. Beating constant velocity on windows **with** a contact is the evidence that NEO learned the collision. |
| `programs`, `codebook_usage_first_step` | Which code sequences are used, overall, with/without contact and per `delta`. Look for a "move" code and a separate "bounce" code. |

`<split>_examples.png` shows six columns per episode: support x, y, prediction,
query x, y, prediction.

## Knobs worth trying

| Setting | Why |
| --- | --- |
| `--set action_codebook_size=4` (default 8) | Fewer codes leave less room to memorise and push toward reusable primitives. |
| `--set length_control_start=1.05 --set length_control_end=1.0` | Anneal the MDL penalty, as in ImageEditing alpha 0.66. |
| `--frame-step 1` or `3` | Smaller or larger time step per primitive. |
| `MODALITY=states` | No vision: tests whether the program/primitive machinery can represent the collision law at all. |
| `--collision-fraction 0.5` on train | More training signal about the contact. |

## Notes

* The simulator is exact for frictionless 1D elastic contact; Box2D's real
  data differs by tiny integration and position-correction effects. The
  converter and evaluation do not depend on which one produced the file.
* Balls not fully inside the frame are excluded from the physics metrics (a
  partly visible ball has a biased centroid).
* High OOD speeds often do not fit in the 10-unit world with 8 frames before
  contact; PhyWorld drops those scenes and so does the simulator.
