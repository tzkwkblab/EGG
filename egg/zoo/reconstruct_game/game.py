# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Builders of the reconstruction game: command-line options -> game and optimizer.

They can be used from Python without going through train.py:

>>> import egg.core as core
>>> opts = core.init(add_game_args(argparse.ArgumentParser()), ["--n_values", "3", "3", "--no_cuda"])
>>> game = build_game(opts, n_inputs=6, n_values=[3, 3])
>>> optimizer = build_optimizer(opts, game)
"""

import argparse
from typing import Optional, Sequence

import egg.core as core
from egg.core.baselines import (
    BuiltInBaseline,
    MeanBaseline,
    MLPCritic,
    NoBaseline,
    SharedFeatureCritic,
)
from egg.core.losses import AttributeReconstructionLoss
from egg.zoo.compo_vs_generalization.archs import PlusOneWrapper
from egg.zoo.reconstruct_game.archs import Receiver, Sender

CRITERIA = ["ce", "gce", "sce"]


def add_game_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("state space and data")
    g.add_argument("--n_values", type=int, nargs="+", default=[4, 4, 4, 4],
                   help="Number of values of each attribute, e.g. `--n_values 10 10 10 10`")
    g.add_argument("--n_train", type=int, default=9000, help="Number of training states")
    g.add_argument("--n_test", type=int, default=1000, help="Number of test states")
    g.add_argument("--split_seed", type=int, default=None,
                   help="Seed of the train/test split (default: --random_seed)")
    g.add_argument("--data_seed", type=int, default=None,
                   help="Seed of the shuffling of training batches (default: --random_seed)")
    g.add_argument("--train_noise_seed", type=int, default=None,
                   help="Seed of the randomness during training, i.e. Sender's sampling and dropout "
                        "(default: --random_seed)")
    g.add_argument("--shuffle_train", dest="shuffle_train", action="store_true", default=True,
                   help="Reshuffle the training set at every pass (default)")
    g.add_argument("--no_shuffle_train", dest="shuffle_train", action="store_false",
                   help="Cycle through the training set in a fixed order")
    g.add_argument("--n_steps", type=int, default=1000, help="Number of training steps (updates)")
    g.add_argument("--steps_per_epoch", type=int, default=100,
                   help="Number of steps the Trainer treats as one epoch")

    g = parser.add_argument_group("agents")
    g.add_argument("--sender_hidden", type=int, default=50)
    g.add_argument("--receiver_hidden", type=int, default=50)
    g.add_argument("--sender_emb", type=int, default=10)
    g.add_argument("--receiver_emb", type=int, default=10)
    g.add_argument("--sender_cell", type=str, default="rnn", choices=["rnn", "gru", "lstm"])
    g.add_argument("--receiver_cell", type=str, default="rnn", choices=["rnn", "gru", "lstm"])
    g.add_argument("--sender_input_layer", type=str, default="linear", choices=["linear", "mlp"])
    g.add_argument("--receiver_dropout", type=float, default=0.0,
                   help="Dropout on the final hidden state of Receiver's RNN")

    g = parser.add_argument_group("losses")
    g.add_argument("--receiver_loss", type=str, default="ce", choices=CRITERIA,
                   help="Loss minimized by Receiver")
    g.add_argument("--sender_loss", type=str, default="same", choices=["same"] + CRITERIA,
                   help="Loss whose negative is Sender's reward (default: same as Receiver's)")
    g.add_argument("--receiver_weight_mode", type=str, default=None,
                   choices=["w", "batch_mean", "relative"],
                   help="Train Receiver with the weighted-CE surrogate of --receiver_loss")
    g.add_argument("--attr_reduction", type=str, default="mean", choices=["mean", "sum"],
                   help="How per-attribute losses are aggregated")
    g.add_argument("--gce_kappa", type=float, default=0.7)
    g.add_argument("--sce_alpha", type=float, default=0.2)
    g.add_argument("--sce_beta", type=float, default=1.0)
    g.add_argument("--sce_A", type=float, default=-4.0)

    g = parser.add_argument_group("optimization (see also --optimizer, --lr, --batch_size)")
    g.add_argument("--sender_lr", type=float, default=None,
                   help="Learning rate of Sender and of the critic (default: --lr)")
    g.add_argument("--receiver_lr", type=float, default=None,
                   help="Learning rate of Receiver (default: --lr)")
    g.add_argument("--weight_decay", type=float, default=None,
                   help="Weight decay (default: the optimizer's own default)")

    g = parser.add_argument_group("reinforce")
    g.add_argument("--sender_entropy_coeff", type=float, default=1e-2)
    g.add_argument("--baseline", type=str, default="mean",
                   choices=["no", "mean", "builtin", "value"])
    g.add_argument("--value_network", type=str, default="separate",
                   choices=["separate", "shared", "shared_detach"],
                   help="With --baseline value: a critic independent from Sender (separate), or one "
                        "that uses the output of Sender's input layer (shared; shared_detach does "
                        "not propagate the critic's loss into that layer)")
    g.add_argument("--value_hidden", type=int, default=None,
                   help="Hidden size of the separate critic (default: --sender_hidden)")
    g.add_argument("--value_layers", type=int, default=1,
                   help="Number of hidden layers of the separate critic")
    g.add_argument("--value_loss_coeff", type=float, default=1.0)
    return parser


def criterion_params(opts) -> dict:
    return {
        "ce": {},
        "gce": {"kappa": opts.gce_kappa},
        "sce": {"alpha": opts.sce_alpha, "beta": opts.sce_beta, "A": opts.sce_A},
    }


def build_loss(
    opts, n_values: Sequence[int], criterion: str, weight_mode: Optional[str] = None
) -> AttributeReconstructionLoss:
    return AttributeReconstructionLoss(
        n_values,
        criterion=criterion,
        criterion_params=criterion_params(opts)[criterion],
        weight_mode=weight_mode,
        attr_reduction=opts.attr_reduction,
        labels_from="labels",
    )


def build_game(
    opts,
    n_inputs: int,
    n_values: Sequence[int],
    train_logging_strategy=None,
    test_logging_strategy=None,
    log_reinforce_stats: bool = False,
) -> core.SenderReceiverRnnReinforce:
    """Creates Sender, Receiver, (optionally) the critic, and the game.

    Modules are created in a fixed order (Sender, Receiver, critic), so that the initial
    parameters depend only on the random seed and not on the losses.
    """
    n_outputs = int(sum(n_values))

    sender_agent = Sender(n_inputs, opts.sender_hidden, opts.sender_input_layer)
    sender = core.RnnSenderReinforce(
        agent=sender_agent,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_emb,
        hidden_size=opts.sender_hidden,
        max_len=opts.max_len,
        cell=opts.sender_cell,
    )
    # symbols 1..vocab_size; 0 is never produced, hence messages have a fixed length
    sender = PlusOneWrapper(sender)

    receiver = core.RnnReceiverDeterministic(
        Receiver(opts.receiver_hidden, n_outputs, opts.receiver_dropout),
        opts.vocab_size + 1,
        opts.receiver_emb,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
    )

    value_fn = None
    if opts.baseline == "value":
        if opts.value_network == "separate":
            value_fn = MLPCritic(
                n_inputs, opts.value_hidden or opts.sender_hidden, opts.value_layers
            )
        else:
            value_fn = SharedFeatureCritic(
                sender_agent,
                opts.sender_hidden,
                detach_input=opts.value_network == "shared_detach",
            )

    loss = build_loss(opts, n_values, opts.receiver_loss, opts.receiver_weight_mode)
    if opts.sender_loss != "same":
        sender_loss = build_loss(opts, n_values, opts.sender_loss)
    elif opts.receiver_weight_mode is not None:
        # the surrogate matches the gradient, not the value: reward with the criterion itself
        sender_loss = build_loss(opts, n_values, opts.receiver_loss)
    else:
        sender_loss = None

    baseline = {
        "no": NoBaseline,
        "mean": MeanBaseline,
        "builtin": BuiltInBaseline,
        "value": NoBaseline,
    }[opts.baseline]

    return core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=baseline,
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
        sender_loss=sender_loss,
        value_fn=value_fn,
        value_loss_coeff=opts.value_loss_coeff,
        log_reinforce_stats=log_reinforce_stats,
    )


def build_optimizer(opts, game):
    """Optimizer with separate parameter groups (and learning rates) for Sender and Receiver."""
    return core.build_game_optimizer(
        game,
        lr=opts.lr,
        sender_lr=opts.sender_lr,
        receiver_lr=opts.receiver_lr,
        weight_decay=opts.weight_decay,
    )
