# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A general attribute-value reconstruction game trained with REINFORCE.

Sender sees an input describing a state (a tuple of categorical attributes), sends a
fixed-length message, and Receiver has to recover the value of every attribute.
See README.md in this directory.
"""

import argparse
import json
import os
import socket
import time

import numpy as np
import torch

import egg.core as core
from egg.core.datasets import CyclicBatchSampler, IndexedTensorLoader
from egg.core.dynamics import DynamicsRecorder, default_record_steps
from egg.core.interaction import LoggingStrategy
from egg.core.util import get_sha
from egg.zoo.reconstruct_game.data import build_state_space, make_sender_inputs
from egg.zoo.reconstruct_game.game import (
    add_game_args,
    build_game,
    build_optimizer,
    criterion_params,
)


def get_params(params):
    parser = argparse.ArgumentParser(description=__doc__)
    add_game_args(parser)
    g = parser.add_argument_group("recording")
    g.add_argument("--out_dir", type=str, default=None,
                   help="Directory for the results (default: nothing is saved, results are printed)")
    g.add_argument("--record_dynamics", action="store_true",
                   help="Record learning dynamics at --record_steps (default: only at the last step)")
    g.add_argument("--record_steps", type=int, nargs="*", default=None,
                   help="Steps to record at (default: 0, 1-2-5 series up to 1000, then every 500)")
    g.add_argument("--n_probe_train", type=int, default=2000,
                   help="Number of training states in the probe set (all test states are included)")
    g.add_argument("--n_policy_samples", type=int, default=8)
    g.add_argument("--topsim_max_pairs_train", type=int, default=100_000,
                   help="Number of pairs used for TopSim on the training part of the probe set")
    g.add_argument("--measure_seed", type=int, default=None,
                   help="Seed of the measurements (default: --random_seed + 2000)")
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--save_checkpoint", action="store_true",
                   help="Save the final parameters to --out_dir")
    opts = core.init(arg_parser=parser, params=params)

    for name in ("split_seed", "data_seed", "train_noise_seed"):
        if getattr(opts, name) is None:
            setattr(opts, name, opts.random_seed)
    if opts.measure_seed is None:
        opts.measure_seed = opts.random_seed + 2000
    assert opts.n_steps % opts.steps_per_epoch == 0, \
        "--n_steps must be a multiple of --steps_per_epoch"
    opts.n_epochs = opts.n_steps // opts.steps_per_epoch
    return opts


def _jsonable(opts) -> dict:
    out = {}
    for k, v in vars(opts).items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


def main(params=None, sender_input_fn=None, batch_transform=None):
    """
    :param sender_input_fn: optional callable (states [N, n], n_values) -> Sender inputs [N, d];
        default is the concatenation of one-hot vectors
    :param batch_transform: optional callable (sender_input, labels) -> sender_input applied to
        every training batch (e.g. per-presentation noise)
    """
    opts = get_params(params)
    device = opts.device
    start_time = time.time()

    # --- data: the split depends only on --split_seed
    train_states, test_states = build_state_space(
        opts.n_values, opts.n_train, opts.n_test, opts.split_seed
    )
    if sender_input_fn is None:
        train_inputs = make_sender_inputs(train_states, opts.n_values)
        test_inputs = make_sender_inputs(test_states, opts.n_values)
    else:
        train_inputs = sender_input_fn(train_states, opts.n_values).float()
        test_inputs = sender_input_fn(test_states, opts.n_values).float()
    train_inputs, train_states = train_inputs.to(device), train_states.to(device)
    test_inputs, test_states = test_inputs.to(device), test_states.to(device)
    n_inputs = train_inputs.size(1)

    # --- game: the initial parameters depend only on --random_seed (set by core.init)
    game = build_game(
        opts,
        n_inputs,
        opts.n_values,
        train_logging_strategy=LoggingStrategy.minimal(),
        test_logging_strategy=LoggingStrategy.minimal(),
        log_reinforce_stats=True,
    )
    optimizer = build_optimizer(opts, game)

    # --- training batches: their order depends only on --data_seed
    sampler = CyclicBatchSampler(
        len(train_inputs),
        opts.batch_size,
        opts.steps_per_epoch,
        shuffle=opts.shuffle_train,
        generator=torch.Generator().manual_seed(opts.data_seed),
    )
    train_loader = IndexedTensorLoader(train_inputs, train_states, sampler, batch_transform)

    # --- recording
    n_probe = min(opts.n_probe_train, len(train_inputs))
    probe_inputs = torch.cat([train_inputs[:n_probe], test_inputs])
    probe_labels = torch.cat([train_states[:n_probe], test_states])
    probe_split = ["train"] * n_probe + ["test"] * len(test_inputs)
    if opts.record_steps is not None:
        record_steps = sorted(set(opts.record_steps) | {opts.n_steps})
    elif opts.record_dynamics:
        record_steps = default_record_steps(opts.n_steps)
    else:
        record_steps = [opts.n_steps]

    out_dir = opts.out_dir
    tmp_dir = None
    if out_dir is None:
        import tempfile

        tmp_dir = tempfile.TemporaryDirectory()
        out_dir = tmp_dir.name
    os.makedirs(out_dir, exist_ok=True)

    config = _jsonable(opts)
    config.update(
        egg_version=get_sha(),
        torch_version=torch.__version__,
        hostname=socket.gethostname(),
        start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_inputs=n_inputs,
        record_steps=record_steps,
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    recorder = DynamicsRecorder(
        game.loss,
        probe_inputs,
        probe_labels,
        probe_split,
        record_steps,
        out_dir,
        vocab_size=opts.vocab_size + 1,
        message_len=opts.max_len,
        robust_criteria=("gce", "sce"),
        criterion_params=criterion_params(opts),
        seed=opts.measure_seed,
        n_policy_samples=opts.n_policy_samples,
        grad_probe_size=n_probe,
        topsim_max_pairs={"train": opts.topsim_max_pairs_train},
        log_every=opts.log_every,
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=None,
        callbacks=[recorder],
    )

    # --- randomness during training (Sender's sampling, dropout) depends only on --train_noise_seed
    torch.manual_seed(opts.train_noise_seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    trainer.train(n_epochs=opts.n_epochs)
    train_seconds = time.time() - train_start

    final = dict(recorder.last_record or {})
    final.update(
        n_steps=opts.n_steps,
        train_seconds=train_seconds,
        measure_seconds_total=recorder.measure_time,
        seconds_per_step=(train_seconds - recorder.measure_time) / opts.n_steps,
        total_seconds=time.time() - start_time,
        max_gpu_memory_mb=(
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        ),
    )
    final = {k: (None if isinstance(v, float) and v != v else v) for k, v in final.items()}
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=1)
    if opts.save_checkpoint and opts.out_dir is not None:
        torch.save(game.state_dict(), os.path.join(out_dir, "checkpoint_final.pt"))

    keys = ["step", "test_topsim", "test_posdis", "test_bosdis", "test_acc", "train_topsim",
            "train_acc", "train_seconds"]
    print(json.dumps({k: final.get(k) for k in keys}), flush=True)
    if tmp_dir is not None:
        tmp_dir.cleanup()
    core.close()
    return final


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
