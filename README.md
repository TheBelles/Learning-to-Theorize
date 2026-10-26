# Theorizing Collisions: Composable Physics Programs from Video

Can a model learn the **physics** of a collision as a small set of reusable
operations, rather than memorising what collisions look like?

Video models trained on PhyWorld's simulated collisions predict well inside
the speeds they were trained on but fail outside them. This project asks
whether a latent program-induction model does better. The model encodes
before/after observations, describes the change between them as a short
program built from a learned codebook of primitive operations, and must reuse
that program on a new collision. If it learns one primitive per time step, it
should handle longer horizons and unseen speeds by composing steps it already
knows.

> **Status:** work in progress. The pipeline is complete; results below will
> be filled in as runs finish.

## Task

Two balls move toward each other horizontally and collide elastically (red:
ball 1, from the left; blue: ball 2, from the right). The data is PhyWorld's
two-ball collision set: 32 frames per video, 0.1 s apart, speeds 1–4 and radii
0.5–1.5 in a 10×10 world.

| | |
| --- | --- |
| **Observation** | Two frames, `t − k` and `t` (`k` = 2 by default), so it shows positions **and** velocities. Pixels: 64×64×6. States: x, y, vx, vy, r for each ball. |
| **Pair** | `x = obs(t)`, `y = obs(t + Δ·k)`: the scene before and after Δ time steps. |
| **Episode** | A support pair from one video and a query pair from a different video, with the same Δ. They share only the physics. |
| **Program** | Up to 4 operations from a codebook of 8 learned primitives. Its length is chosen by a minimum-description-length rule. |
| **Test** | The program (and its length) is inferred on the support pair and executed on the query pair. |

Generalisation is measured on videos never seen in training:

- **ID:** same speeds and radii as training, with and without the contact inside the window;
- **Length OOD:** Δ = 5–8 when training used Δ = 1–4;
- **Velocity OOD:** speeds outside [1, 4].

## Setup

Python 3.10, PyTorch 2.7.1, CUDA 12.6 (pinned environment in
[environments/README.md](environments/README.md)), then:

```bash
pip install -e .
pip install imageio imageio-ffmpeg huggingface_hub
```

Check the installation on CPU (a few minutes, tiny simulated data):

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_phyworld_collision.py
SMOKE=1 bash scripts/phyworld_collision.sh
```

## Run

The whole pipeline:

```bash
bash scripts/phyworld_collision.sh                  # pixels
MODALITY=states bash scripts/phyworld_collision.sh  # state vectors (fast, start here)
```

It runs these stages in order:

1. **Data.** It downloads `collision_30K.hdf5` from [PhyWorld on Hugging Face](https://huggingface.co/datasets/magicr/phyworld) and simulates an OOD-speed set in the same format.
2. **Episodes.** It converts videos into support/query episodes, with separate train, ID-test, length-OOD and velocity-OOD files.
3. **Observation model.** It pretrains a VAE that is frozen afterwards. Check `reconstructions_epoch*.png` to see that both balls are sharp.
4. **Theorizer.** It trains the program-induction model; `checkpoints/best.pt` is selected by ID query error.
5. **Evaluation.** It computes transfer error, program length vs. Δ, ball position and velocity errors against constant-velocity and copy baselines, and which primitives are used.

Every stage can also be run on its own, and every training setting can be
changed with `--set name=value`. See
[docs/phyworld_collision.md](docs/phyworld_collision.md) for the commands,
options and how to read the results.

## Results

To be filled in.

| Split | Query error | Program length = Δ | Velocity error (all) | Velocity error (with contact) | Constant-velocity baseline (with contact) |
| --- | --- | --- | --- | --- | --- |
| ID | | | | | |
| Length OOD (Δ 5–8) | | | | | |
| Velocity OOD | | | | | |

Errors are in world units (the world is 10 units wide).

## Code

| Path | Contents |
| --- | --- |
| `src/tasks/phyworld_collision/physics.py` | Constants, rendering, closed-form collision simulator, ball tracking |
| `src/tasks/phyworld_collision/data/` | PhyWorld file reader, OOD simulator, episode converter, datasets |
| `src/tasks/phyworld_collision/models/vae.py` | Observation models for stacked frames and state vectors |
| `src/tasks/phyworld_collision/objective.py` | Foreground-weighted losses and metrics |
| `src/tasks/phyworld_collision/observation_pretraining.py` | VAE pretraining |
| `src/tasks/phyworld_collision/theorizer_runner.py` | Training |
| `src/tasks/phyworld_collision/evaluation.py` | Evaluation and physics metrics |
| `scripts/phyworld_collision.sh` | End-to-end pipeline |
| `tests/test_phyworld_collision.py` | Unit tests and a CPU end-to-end run |

## Team

_Add team members here._

## Acknowledgements and license

This project is built on the open-source
[Learning to Theorize](https://github.com/ahn-ml/learning-to-theorize)
codebase (NEO model, theory programmer, executor and quantizer) and uses the
[PhyWorld](https://github.com/phyworld/phyworld) collision data and generator
settings.

The code is under the MIT License ([LICENSE](LICENSE)). The vector-quantizer
files remain under the Apache License 2.0; see
[third-party notices](docs/third-party-notices.md).
