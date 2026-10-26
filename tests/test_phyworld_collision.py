"""PhyWorld collision task: physics, data, models and a CPU end-to-end run.

Run with ``CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_phyworld_collision.py``.
The end-to-end test takes a minute or two on CPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tasks.phyworld_collision import physics as P
from tasks.phyworld_collision.data import episodes as E
from tasks.phyworld_collision.data.simulate import preset_combinations, simulate_videos


@pytest.fixture(scope="module")
def videos():
    rng = np.random.default_rng(0)
    combinations = preset_combinations("tiny", rng)
    return simulate_videos(combinations, rng)


# ---------------------------------------------------------------------------
# NumPy only
# ---------------------------------------------------------------------------


def test_elastic_collision_conserves_momentum_and_energy(videos):
    positions, init = videos
    for p, (r1, r2, v1, v2) in zip(positions[:20], init[:20]):
        m1, m2 = P.masses(r1, r2)
        velocity = np.diff(p[:, :, 0], axis=0) / P.FRAME_DT
        before, after = velocity[0], velocity[-1]
        assert np.isclose(m1 * before[0] + m2 * before[1], m1 * after[0] + m2 * after[1])
        assert np.isclose(m1 * before[0] ** 2 + m2 * before[1] ** 2, m1 * after[0] ** 2 + m2 * after[1] ** 2)
        assert P.collision_frame(p, np.array([r1, r2])) >= P.MIN_PRE_FRAMES - 1


def test_slow_contacts_are_inelastic_like_box2d():
    w1, w2 = P.post_collision_velocities(1.0, 1.0, 0.3, -0.2, inelastic_threshold=1.0)
    assert np.isclose(w1, w2) and np.isclose(w1, 0.05)


def test_rendering_and_tracking_agree(videos):
    positions, init = videos
    frames = P.render_frames(positions[0], init[0, :2], 64)
    track = P.ball_centroids(frames)
    inside = np.ones(positions.shape[1], dtype=bool)
    for ball in range(2):
        r = init[0, ball]
        inside &= (positions[0, :, ball, 0] - r >= 0) & (positions[0, :, ball, 0] + r <= P.WORLD_SCALE)
    error = np.linalg.norm(track.centers[inside] - positions[0][inside], axis=-1)
    assert error.max() < 0.05


def test_episode_shapes_and_collision_control(videos):
    positions, init = videos
    rng = np.random.default_rng(1)
    source = E.rendered_frame_source(positions, init[:, :2], 64)
    record = E.build_episode(
        rng, support_video=0, query_video=1, delta=3, frame_step=2, positions=positions,
        init=init, first_index=0, frame_source=source, collision_fraction=1.0,
    )
    assert record.grids.shape == (4, 64, 64, 6) and record.grids.dtype == np.uint8
    assert record.states.shape == (4, P.STATE_DIM)
    assert record.collides.all()
    physical = P.state_to_physical(record.states)
    assert np.allclose(physical["position"], record.positions[:, 1], atol=1e-4)
    with pytest.raises(ValueError):
        E.valid_start_range(20, 2, 32)


def test_video_split_is_disjoint_and_classification():
    train = E.video_split_mask(500, "train", test_fraction=0.1, split_seed=0)
    test = E.video_split_mask(500, "test", test_fraction=0.1, split_seed=0)
    assert not (train & test).any() and (train | test).all()
    labels = E.classify_videos(np.array([[1, 1, 2, 2], [1, 1, 5, 2], [1.8, 1, 2, 2], [0.3, 1, 0.1, 2]]))
    assert list(labels) == ["id", "ood-velocity", "ood-radius", "ood-both"]


# ---------------------------------------------------------------------------
# HDF5 + PyTorch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def episode_files(tmp_path_factory, videos):
    pytest.importorskip("h5py")
    from tasks.phyworld_collision.data import convert
    from tasks.phyworld_collision.data.phyworld_hdf5 import write_phyworld_file

    root = tmp_path_factory.mktemp("collision")
    positions, init = videos
    write_phyworld_file(root / "source.hdf5", positions, init, chunk_size=64)
    files = {}
    for name, videos_split, deltas, count in (
        ("train", "train", "1,2,3,4", 96),
        ("test", "test", "1,2,3,4", 24),
        ("length", "test", "5,6", 16),
    ):
        arguments = convert.build_parser().parse_args(
            ["--source", str(root / "source.hdf5"), "--output", str(root / f"{name}.h5"),
             "--episodes", str(count), "--deltas", deltas, "--videos", videos_split,
             "--test-fraction", "0.3"]
        )
        summary = convert.convert(arguments)
        assert summary["episodes"] == count
        files[name] = root / f"{name}.h5"
    return root, files


def test_converted_files(episode_files):
    torch = pytest.importorskip("torch")
    from tasks.phyworld_collision.data.dataset import EpisodeDataset, ObservationDataset, load_metadata

    _, files = episode_files
    dataset = EpisodeDataset(files["train"], "pixels")
    observations, index = dataset[3]
    assert observations.shape == (4, 64, 64, 6) and observations.dtype == torch.uint8 and index == 3
    states, _ = EpisodeDataset(files["train"], "states")[3]
    assert states.shape == (4, P.STATE_DIM)
    assert len(ObservationDataset(files["train"], "pixels")) == 4 * len(dataset)
    metadata = load_metadata(files["length"])
    assert set(np.unique(metadata.delta)) <= {5, 6}


@pytest.mark.parametrize("modality", ["pixels", "states"])
def test_neo_forward_train_and_eval(modality, episode_files):
    torch = pytest.importorskip("torch")
    from tasks.phyworld_collision.data.dataset import EpisodeDataset, collate_episodes
    from tasks.phyworld_collision.experiment_config import Experiment, TrainingConfig
    from tasks.phyworld_collision.models.vae import ObservationConfig
    from tasks.phyworld_collision.task import build_neo

    _, files = episode_files
    observation = ObservationConfig(modality=modality, state_dim=32, base_channels=16, bottleneck_channels=32)
    training = TrainingConfig(state_dim=32, policy_hidden_dim=32, transition_hidden_dim=32, batch_size=4)
    model = build_neo(Experiment(training), observation)
    batch = collate_episodes([EpisodeDataset(files["train"], modality)[i] for i in range(4)])
    model.train()
    output = model(batch.observations, is_eval=False)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    model.eval()
    with torch.no_grad():
        output = model(batch.observations, is_eval=True, num_transitions=6)
    assert output.query_metrics is not None and output.selected_lengths.shape == (8,)
    assert torch.equal(output.selected_lengths[::2], output.selected_lengths[1::2])


def test_end_to_end_cpu(episode_files, tmp_path):
    pytest.importorskip("torch")
    from tasks.phyworld_collision import evaluation, observation_pretraining, theorizer_runner

    _, files = episode_files
    observation_run = observation_pretraining.run(
        observation_pretraining.build_parser().parse_args(
            ["--train-h5", str(files["train"]), "--test-h5", str(files["test"]),
             "--output-root", str(tmp_path / "observation"), "--epochs", "1", "--batch-size", "32",
             "--num-workers", "0", "--state-dim", "32", "--base-channels", "16",
             "--bottleneck-channels", "32", "--cpu"]
        )
    )
    observation_checkpoint = observation_run / "checkpoints" / "best.pt"
    assert observation_checkpoint.exists()
    assert list(observation_run.glob("reconstructions_epoch*.png"))

    neo_run = theorizer_runner.run(
        theorizer_runner.build_parser().parse_args(
            ["--train-h5", str(files["train"]), "--test-h5", str(files["test"]),
             "--observation-checkpoint", str(observation_checkpoint),
             "--output-root", str(tmp_path / "neo"), "--cpu",
             "--set", "epochs=1", "--set", "batch_size=16", "--set", "num_workers=0",
             "--set", "policy_hidden_dim=32", "--set", "transition_hidden_dim=32"]
        )
    )
    neo_checkpoint = neo_run / "checkpoints" / "best.pt"
    assert neo_checkpoint.exists()
    assert json.loads((neo_run / "result.json").read_text())["status"] == "completed"

    output_dir = tmp_path / "evaluation"
    evaluation.main(
        ["--checkpoint", str(neo_checkpoint), "--output-dir", str(output_dir), "--cpu",
         "--batch-size", "8", "--split", f"id={files['test']}", "--split", f"length={files['length']}"]
    )
    report = json.loads((output_dir / "evaluation.json").read_text())
    assert report["length"]["num_transitions"] == 6
    assert report["id"]["all"]["episodes"] == 24
    assert (output_dir / "id_examples.png").exists()
