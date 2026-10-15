# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
import torch

from egg.zoo.reconstruct_game.train import main

BASE = ["--n_values", "5", "5", "--vocab_size", "5", "--max_len", "3", "--n_train", "20",
        "--n_test", "5", "--batch_size", "16", "--steps_per_epoch", "20", "--no_cuda",
        "--sender_hidden", "32", "--receiver_hidden", "32", "--sender_cell", "gru",
        "--receiver_cell", "gru", "--receiver_dropout", "0.5", "--optimizer", "adamw",
        "--lr", "0.01", "--baseline", "value", "--n_probe_train", "20", "--save_checkpoint"]


def _run(tmp_path, name, extra, n_steps=60, seed=1):
    out = tmp_path / name
    final = main(BASE + ["--n_steps", str(n_steps), "--random_seed", str(seed),
                         "--out_dir", str(out)] + list(extra))
    protocol = np.load(out / "protocol.npz")
    ckpt = torch.load(out / "checkpoint_final.pt", weights_only=True)
    return final, protocol, ckpt, out


def _equal_ckpt(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_learns_and_writes_outputs(tmp_path):
    final, protocol, _, out = _run(tmp_path, "a", ["--record_dynamics"], n_steps=400)
    assert final["train_acc_or"] > 0.4  # chance is 0.2
    for name in ("config.json", "dynamics.jsonl", "train_log.jsonl", "final.json", "protocol.npz"):
        assert (out / name).exists()
    rows = [json.loads(line) for line in open(out / "dynamics.jsonl")]
    assert [r["step"] for r in rows] == [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]
    assert protocol["greedy_messages"].shape == (10, 25, 3)
    assert protocol["mi"].shape == (10, 2, 3)
    assert rows[0]["probe_w_gce_mean"] < 0.5  # about (1/5)^0.7 = 0.32 before training
    assert "batch_q_mean" in rows[-1] and "probe_grad_cos_gce" in rows[-1]


def test_same_seed_is_reproducible(tmp_path):
    f1, p1, c1, _ = _run(tmp_path, "a", ["--receiver_loss", "gce"])
    f2, p2, c2, _ = _run(tmp_path, "b", ["--receiver_loss", "gce"])
    assert _equal_ckpt(c1, c2)
    assert np.array_equal(p1["greedy_messages"], p2["greedy_messages"])
    assert f1["test_topsim"] == f2["test_topsim"]


def test_recording_does_not_change_training(tmp_path):
    _, _, c1, _ = _run(tmp_path, "a", [])
    _, _, c2, _ = _run(tmp_path, "b", ["--record_dynamics"])
    assert _equal_ckpt(c1, c2)


def test_losses_share_initial_state_and_data(tmp_path):
    steps = ["--record_steps", "0", "1"]
    _, p_ce, c_ce, _ = _run(tmp_path, "ce", steps + ["--receiver_loss", "ce"])
    _, p_gce, c_gce, _ = _run(tmp_path, "gce", steps + ["--receiver_loss", "gce"])
    # same initial protocol, same probe set (hence same split)
    assert np.array_equal(p_ce["greedy_messages"][0], p_gce["greedy_messages"][0])
    assert np.array_equal(p_ce["mi"][0], p_gce["mi"][0])
    assert np.array_equal(p_ce["probe_states"], p_gce["probe_states"])
    assert not _equal_ckpt(c_ce, c_gce)
    # the first training batch has the same inputs and the same sampled messages:
    # the confidences on it are identical, whatever loss is minimized afterwards
    rows = [[json.loads(line) for line in open(tmp_path / n / "dynamics.jsonl")] for n in ("ce", "gce")]
    assert rows[0][1]["batch_q_mean"] == rows[1][1]["batch_q_mean"]


def test_train_noise_seed_only_changes_training_noise(tmp_path):
    steps = ["--record_steps", "0"]
    _, p1, c1, _ = _run(tmp_path, "a", steps + ["--train_noise_seed", "1"])
    _, p2, c2, _ = _run(tmp_path, "b", steps + ["--train_noise_seed", "1001"])
    assert np.array_equal(p1["greedy_messages"][0], p2["greedy_messages"][0])
    assert np.array_equal(p1["probe_states"], p2["probe_states"])
    assert not _equal_ckpt(c1, c2)


def test_options_run(tmp_path):
    variants = {
        "shared": ["--value_network", "shared"],
        "detach": ["--value_network", "shared_detach"],
        "noshuffle": ["--no_shuffle_train"],
        "mean": ["--baseline", "mean", "--sender_loss", "ce", "--receiver_loss", "sce"],
        "mlp": ["--sender_input_layer", "mlp", "--attr_reduction", "sum"],
        "rel": ["--receiver_loss", "gce", "--receiver_weight_mode", "relative",
                "--receiver_lr", "0.001"],
    }
    for name, extra in variants.items():
        final, _, _, _ = _run(tmp_path, name, extra, n_steps=20)
        assert final["step"] == 20


def test_heterogeneous_values(tmp_path):
    out = tmp_path / "h"
    final = main(["--n_values", "2", "3", "4", "--vocab_size", "4", "--max_len", "3",
                  "--n_train", "18", "--n_test", "6", "--batch_size", "8", "--n_steps", "20",
                  "--steps_per_epoch", "10", "--no_cuda", "--out_dir", str(out)])
    assert final["step"] == 20
