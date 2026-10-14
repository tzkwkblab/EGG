# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch

import egg.core as core
from egg.zoo.reconstruct_game.data import build_state_space, one_hot_states
from egg.zoo.reconstruct_game.game import build_game, build_loss, build_optimizer
from egg.zoo.reconstruct_game.train import get_params

N_VALUES = [4, 3, 5]
BASE = ["--n_values", "4", "3", "5", "--vocab_size", "5", "--max_len", "3", "--no_cuda",
        "--sender_hidden", "16", "--receiver_hidden", "16", "--sender_cell", "gru",
        "--receiver_cell", "gru", "--random_seed", "3"]


def _setup(extra=()):
    opts = get_params(BASE + list(extra))
    states, _ = build_state_space(N_VALUES, 40, 10, 0)
    inputs = one_hot_states(states, N_VALUES)
    game = build_game(opts, inputs.size(1), N_VALUES, log_reinforce_stats=True)
    return opts, game, inputs, states


def _grads(game, inputs, states, seed=11):
    game.zero_grad()
    torch.manual_seed(seed)
    loss, interaction = game(inputs, states)
    loss.backward()
    return {n: p.grad.clone() for n, p in game.named_parameters() if p.grad is not None}, loss, interaction


def _same(a, b, prefix):
    keys = [k for k in a if k.startswith(prefix)]
    assert keys
    return all(torch.allclose(a[k], b[k], atol=1e-7) for k in keys)


def test_defaults_are_unchanged_by_logging_stats():
    opts, game, inputs, states = _setup()
    plain = build_game(opts, inputs.size(1), N_VALUES)
    plain.load_state_dict(game.state_dict())
    _, loss_a, inter_a = _grads(game, inputs, states)
    _, loss_b, inter_b = _grads(plain, inputs, states)
    assert torch.equal(loss_a, loss_b)
    assert "reward_loss" in inter_a.aux and "reward_loss" not in inter_b.aux


def test_sender_loss_is_separated_from_receiver_loss():
    opts, both_ce, inputs, states = _setup()
    recv_gce = copy.deepcopy(both_ce)
    recv_gce.loss = build_loss(opts, N_VALUES, "gce")
    recv_gce.sender_loss = build_loss(opts, N_VALUES, "ce")
    both_gce = copy.deepcopy(both_ce)
    both_gce.loss = build_loss(opts, N_VALUES, "gce")

    g_ce, _, _ = _grads(both_ce, inputs, states)
    g_mixed, _, _ = _grads(recv_gce, inputs, states)
    g_gce, _, _ = _grads(both_gce, inputs, states)

    # Receiver follows `loss` only, Sender follows `sender_loss` only
    assert _same(g_mixed, g_gce, "receiver.")
    assert _same(g_mixed, g_ce, "sender.")
    assert not _same(g_mixed, g_ce, "receiver.")
    assert not _same(g_mixed, g_gce, "sender.")


def test_surrogate_loss_rewards_sender_with_the_criterion():
    _, game, inputs, states = _setup(["--receiver_loss", "gce", "--receiver_weight_mode", "w"])
    _, ref, _, _ = _setup(["--receiver_loss", "gce"])
    ref.load_state_dict(game.state_dict())
    g, _, _ = _grads(game, inputs, states)
    g_ref, _, _ = _grads(ref, inputs, states)
    assert _same(g, g_ref, "receiver.") and _same(g, g_ref, "sender.")


def _value_grads(network, coeff):
    _, game, inputs, states = _setup(
        ["--baseline", "value", "--value_network", network, "--value_loss_coeff", str(coeff)]
    )
    return _grads(game, inputs, states), game, inputs


def test_separate_critic_does_not_touch_sender():
    (g1, _, inter), game, inputs = _value_grads("separate", 1.0)
    (g10, _, _), _, _ = _value_grads("separate", 10.0)
    assert _same(g1, g10, "sender.") and _same(g1, g10, "receiver.")
    assert not _same(g1, g10, "value_fn.")

    # the critic is trained by regression only: no gradient from the policy loss
    game.zero_grad()
    value = game.value_fn(inputs)
    (value - inter.aux["reward_loss"]).pow(2).mean().backward()
    for name, p in game.value_fn.named_parameters():
        assert torch.allclose(p.grad, g1["value_fn." + name], atol=1e-6)
    assert torch.allclose(inter.aux["baseline"], value.detach())


def test_shared_critic_gradient_into_the_encoder():
    (g1, _, _), _, _ = _value_grads("shared", 1.0)
    (g10, _, _), _, _ = _value_grads("shared", 10.0)
    enc = "sender.wrapped.agent.fc1."
    assert not _same(g1, g10, enc)

    (d1, _, _), _, _ = _value_grads("shared_detach", 1.0)
    (d10, _, _), _, _ = _value_grads("shared_detach", 10.0)
    assert _same(d1, d10, enc)
    assert not _same(d1, d10, "value_fn.head.")


def test_game_optimizer_param_groups():
    opts, game, _, _ = _setup(
        ["--baseline", "value", "--optimizer", "adamw", "--lr", "0.001",
         "--receiver_lr", "0.0001", "--weight_decay", "0.05"]
    )
    opt = build_optimizer(opts, game)
    assert isinstance(opt, torch.optim.AdamW)
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["sender"]["lr"] == 0.001 and groups["receiver"]["lr"] == 0.0001
    assert all(g["weight_decay"] == 0.05 for g in opt.param_groups)
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(game.parameters()))
    n_recv = len(list(game.receiver.parameters()))
    assert len(groups["receiver"]["params"]) == n_recv

    opts, game, _, _ = _setup()
    opt = build_optimizer(opts, game)  # defaults: Adam, the optimizer's own weight decay
    assert isinstance(opt, torch.optim.Adam) and opt.param_groups[0]["weight_decay"] == 0
    assert isinstance(core.build_optimizer(game.parameters()), torch.optim.Adam)
