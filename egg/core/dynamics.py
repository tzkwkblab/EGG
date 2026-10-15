# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Step-level recording of learning dynamics in attribute reconstruction games.

`DynamicsRecorder` is a callback that, at a given list of training steps, measures the state of
the Receiver (confidence in the correct value, gradient weights implied by noise-robust losses,
how the gradients of different losses compare), of the Sender (reward, advantage, policy entropy)
and of the protocol (compositionality, accuracy, how many messages changed, the mutual
information between attributes and message positions) on a fixed probe set of inputs.

Measurements do not advance the random number stream used by training: the global RNG state is
saved before and restored after each measurement.
"""

import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .callbacks import Callback
from .interaction import Interaction
from .language_analysis import (
    compute_bosdis,
    compute_posdis,
    compute_topsim_from_pairs,
    mutual_info_matrix,
    sample_pairs,
)
from .losses import ROBUST_DEFAULTS, loss_from_log_prob, robust_weight
from .util import find_lengths

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def default_record_steps(n_steps: int, dense_until: int = 1000, every: int = 500):
    """0, then a 1-2-5 series up to `dense_until`, then every `every` steps, and the last step.

    >>> default_record_steps(2000)
    [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
    """
    steps = {0, n_steps}
    base = 1
    while base <= dense_until:
        for m in (1, 2, 5):
            if m * base <= min(dense_until, n_steps):
                steps.add(m * base)
        base *= 10
    steps.update(range(dense_until + every, n_steps + 1, every))
    return sorted(steps)


@contextmanager
def preserved_rng(seed: int):
    """Runs the block with the global torch RNG seeded with `seed`, then restores its state."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def module_modes(game: nn.Module, sample: bool):
    """Temporarily puts the game into a measurement mode and restores all training flags.

    sample=False: everything in eval mode (Sender outputs greedy messages).
    sample=True:  train mode (Sender samples; cuDNN RNNs allow backward) but Dropout disabled.
    """
    saved = {m: m.training for m in game.modules()}
    try:
        if sample:
            game.train()
            for m in game.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                    m.eval()
        else:
            game.eval()
        yield
    finally:
        for m, flag in saved.items():
            m.training = flag


def _stats(x: torch.Tensor, prefix: str, quantiles: bool = False) -> Dict[str, float]:
    x = x.detach().float().flatten()
    out = {f"{prefix}_mean": x.mean().item(), f"{prefix}_std": x.std().item()}
    if quantiles:
        q = torch.quantile(x.cpu(), torch.tensor(QUANTILES))
        for level, value in zip(QUANTILES, q.tolist()):
            out[f"{prefix}_q{int(level * 100):02d}"] = value
    return out


def _weight_stats(q_y: torch.Tensor, criteria: Sequence[str], params: Dict, prefix: str):
    out = {}
    for c in criteria:
        w = robust_weight(q_y, c, **params.get(c, {})).float().flatten()
        mean = w.mean().item()
        out[f"{prefix}_w_{c}_mean"] = mean
        out[f"{prefix}_w_{c}_cv"] = (w.std() / w.mean()).item() if mean > 0 else float("nan")
    return out


def _clean(d: Dict) -> Dict:
    """Makes a dict JSON-serializable (nan -> None)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and v != v:
            v = None
        out[k] = v
    return out


class DynamicsRecorder(Callback):
    """
    :param attr_loss: an `AttributeReconstructionLoss`-like object (provides `log_target_prob`,
        `reduce`, `n_values`); used to compute confidences and the losses compared in the gradient probe
    :param probe_inputs: Sender inputs of the probe set, [P, d]
    :param probe_labels: attribute values of the probe set, [P, n]
    :param probe_split: [P] array of strings ("train"/"test"); metrics are computed per split
    :param record_steps: training steps (number of updates done) at which to record; 0 = before training
    :param out_dir: directory for `dynamics.jsonl`, `train_log.jsonl`, `protocol.npz`
    :param vocab_size: number of symbols including the unused/eos symbol 0 (for BosDis)
    :param message_len: number of leading positions of a message that carry symbols (a trailing
        constant/eos position is dropped from the analyses)
    :param robust_criteria: criteria whose gradient weights and gradients are compared with CE
    :param criterion_params: {criterion: {param: value}}, overrides of the defaults
    :param seed: seed of the measurements
    :param n_policy_samples: messages sampled per probe input for the mutual information matrix
    :param grad_probe_split, grad_probe_size: probe inputs used for the gradient comparison
    :param topsim_max_pairs: {split: max number of pairs} (default: all pairs)
    :param log_every: period (in steps) of the light-weight training log
    """

    def __init__(
        self,
        attr_loss,
        probe_inputs: torch.Tensor,
        probe_labels: torch.Tensor,
        probe_split: Sequence[str],
        record_steps: Sequence[int],
        out_dir: str,
        vocab_size: int,
        message_len: int,
        robust_criteria: Sequence[str] = ("gce", "sce"),
        criterion_params: Optional[Dict[str, Dict[str, float]]] = None,
        seed: int = 0,
        n_policy_samples: int = 8,
        grad_probe_split: str = "train",
        grad_probe_size: int = 2000,
        topsim_max_pairs: Optional[Dict[str, int]] = None,
        log_every: int = 50,
    ):
        self.attr_loss = attr_loss
        self.probe_inputs = probe_inputs
        self.probe_labels = probe_labels.long()
        self.probe_split = np.asarray(probe_split)
        self.record_steps = set(int(s) for s in record_steps)
        self.out_dir = out_dir
        self.vocab_size = vocab_size
        self.message_len = message_len
        self.robust_criteria = list(robust_criteria)
        self.criterion_params = {c: dict(ROBUST_DEFAULTS[c]) for c in self.robust_criteria}
        for c, p in (criterion_params or {}).items():
            if c in self.criterion_params:
                self.criterion_params[c].update(p)
        self.seed = seed
        self.n_policy_samples = n_policy_samples
        self.log_every = log_every

        self.splits = sorted(set(self.probe_split.tolist()))
        self.split_idx = {s: np.nonzero(self.probe_split == s)[0] for s in self.splits}
        topsim_max_pairs = topsim_max_pairs or {}
        self.pairs = {
            s: sample_pairs(len(idx), topsim_max_pairs.get(s), seed=seed)
            for s, idx in self.split_idx.items()
        }
        grad_idx = self.split_idx.get(grad_probe_split, np.arange(len(self.probe_split)))
        self.grad_idx = torch.as_tensor(grad_idx[:grad_probe_size])

        self.step = 0
        self.trainer = None
        self.prev_greedy = None
        self.rec_steps: List[int] = []
        self.rec_greedy: List[np.ndarray] = []
        self.rec_mi: List[np.ndarray] = []
        self.last_record: Optional[Dict] = None
        self.measure_time = 0.0

        os.makedirs(out_dir, exist_ok=True)
        self._dyn = open(os.path.join(out_dir, "dynamics.jsonl"), "w")
        self._log = open(os.path.join(out_dir, "train_log.jsonl"), "w")

    # ----- helpers -------------------------------------------------------------------------

    @property
    def game(self):
        return self.trainer.game

    def _send(self, inputs):
        message, _, _ = self.game.sender(inputs, None)
        return message

    def _receive(self, message):
        output, _, _ = self.game.receiver(message, None, None, find_lengths(message))
        return output

    def _symbols(self, message: torch.Tensor) -> np.ndarray:
        return message[:, : self.message_len].cpu().numpy()

    # ----- measurements --------------------------------------------------------------------

    def _train_batch_stats(self, logs: Interaction) -> Dict:
        aux = logs.aux
        out = {}
        if "q_y" in aux:
            out.update(_stats(aux["q_y"], "batch_q", quantiles=True))
            out.update(
                _weight_stats(aux["q_y"], self.robust_criteria, self.criterion_params, "batch")
            )
        if "acc" in aux:
            out["batch_acc"] = aux["acc"].float().mean().item()
        if "reward_loss" in aux:
            out.update(_stats(aux["reward_loss"], "batch_reward_loss"))
            out.update(_stats(aux["reward_loss"] - aux["baseline"], "batch_advantage"))
        if "sender_entropy" in aux:
            ent = aux["sender_entropy"].float().mean(dim=0)[: self.message_len]
            out["batch_entropy_per_position"] = ent.tolist()
            out["batch_entropy_mean"] = ent.mean().item()
        return out

    def _protocol(self, step: int) -> Dict:
        """Greedy protocol on the probe set: compositionality, accuracy, change rate."""
        out = {}
        labels = self.probe_labels
        with torch.no_grad(), module_modes(self.game, sample=False):
            message = self._send(self.probe_inputs)
            log_q, pred = self.attr_loss.log_target_prob(self._receive(message), labels)
        greedy = self._symbols(message)
        correct = (pred == labels).cpu().numpy()
        attrs = labels.cpu().numpy()

        for s, idx in self.split_idx.items():
            a, m = attrs[idx], greedy[idx]
            out[f"{s}_topsim"] = compute_topsim_from_pairs(a, m, self.pairs[s])
            out[f"{s}_posdis"] = compute_posdis(a, m)
            out[f"{s}_bosdis"] = compute_bosdis(a, m, self.vocab_size)
            out[f"{s}_acc"] = float(correct[idx].all(axis=1).mean())
            out[f"{s}_acc_or"] = float(correct[idx].mean())
            out[f"{s}_n_distinct_messages"] = int(len({tuple(r) for r in m.tolist()}))
            out[f"{s}_greedy_q_mean"] = float(log_q[torch.as_tensor(idx)].exp().mean().item())
            if self.prev_greedy is not None:
                changed = (m != self.prev_greedy[idx]).any(axis=1)
                out[f"{s}_change_rate"] = float(changed.mean())
        self.prev_greedy = greedy
        self.rec_greedy.append(greedy.astype(np.int16))
        return out

    def _latent_structure(self) -> Dict:
        """I[attribute; symbol at position] under the stochastic policy."""
        with torch.no_grad(), module_modes(self.game, sample=True):
            samples = [
                self._symbols(self._send(self.probe_inputs))
                for _ in range(self.n_policy_samples)
            ]
        attrs = self.probe_labels.cpu().numpy()
        mi = mutual_info_matrix(
            np.concatenate([attrs] * self.n_policy_samples), np.concatenate(samples)
        )
        self.rec_mi.append(mi.astype(np.float32))
        return {"policy_mi_total": float(mi.sum()), "policy_mi_max_per_position": mi.max(axis=0).tolist()}

    def _gradient_probe(self) -> Dict:
        """Sampled messages on (part of) the probe set: confidence, gradient weights, their relation
        to exploration, and the comparison of the Receiver gradients of CE and of the robust losses."""
        out = {}
        idx = self.grad_idx.to(self.probe_inputs.device)
        inputs, labels = self.probe_inputs[idx], self.probe_labels[idx]

        with torch.no_grad(), module_modes(self.game, sample=False):
            greedy = self._send(inputs)

        with module_modes(self.game, sample=True):
            with torch.no_grad():
                message = self._send(inputs)
            log_q, _ = self.attr_loss.log_target_prob(self._receive(message), labels)

            params = [p for p in self.game.receiver.parameters() if p.requires_grad]
            grads = {}
            for c in ["ce"] + self.robust_criteria:
                per_attr = loss_from_log_prob(log_q, c, **self.criterion_params.get(c, {}))
                loss = self.attr_loss.reduce(per_attr).mean()
                g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                grads[c] = torch.cat([x.flatten() for x in g if x is not None])

        q_y = log_q.detach().exp()
        out.update(_stats(q_y, "probe_q", quantiles=True))
        out.update(_weight_stats(q_y, self.robust_criteria, self.criterion_params, "probe"))

        ce_norm = grads["ce"].norm()
        out["probe_grad_norm_ce"] = ce_norm.item()
        for c in self.robust_criteria:
            g = grads[c]
            out[f"probe_grad_cos_{c}"] = torch.nn.functional.cosine_similarity(
                g, grads["ce"], dim=0
            ).item()
            out[f"probe_grad_norm_ratio_{c}"] = (g.norm() / ce_norm).item()

        # exploration: does the sampled message deviate from the greedy one?
        same = (message[:, : self.message_len] == greedy[:, : self.message_len]).all(dim=1)
        out["probe_greedy_match_rate"] = same.float().mean().item()
        for c in self.robust_criteria:
            w = robust_weight(q_y, c, **self.criterion_params.get(c, {})).mean(dim=1)
            for name, mask in (("match", same), ("deviate", ~same)):
                value = w[mask].mean().item() if mask.any() else float("nan")
                out[f"probe_w_{c}_{name}"] = value
        for name, mask in (("match", same), ("deviate", ~same)):
            out[f"probe_q_{name}"] = q_y[mask].mean().item() if mask.any() else float("nan")
        return out

    def record(self, step: int, logs: Optional[Interaction]) -> None:
        import time

        start = time.time()
        row = {"step": step}
        if logs is not None:
            row.update(self._train_batch_stats(logs))
        with preserved_rng(self.seed * 1_000_003 + step):
            row.update(self._protocol(step))
            row.update(self._latent_structure())
            row.update(self._gradient_probe())
        self.rec_steps.append(step)
        self.measure_time += time.time() - start
        row["measure_seconds"] = time.time() - start
        self.last_record = row
        self._dyn.write(json.dumps(_clean(row)) + "\n")
        self._dyn.flush()

    # ----- callback interface --------------------------------------------------------------

    def on_train_begin(self, trainer_instance):
        self.trainer = trainer_instance
        if 0 in self.record_steps:
            self.record(0, None)

    def on_batch_end(self, logs: Interaction, loss, batch_id: int, is_training: bool = True):
        if not is_training:
            return
        self.step += 1
        if self.log_every and self.step % self.log_every == 0:
            aux = logs.aux
            row = {"step": self.step, "loss": float(loss)}
            for key in ("acc", "acc_or", "reward_loss", "baseline", "q_y"):
                if key in aux:
                    row[key] = aux[key].float().mean().item()
            if "sender_entropy" in aux:
                row["sender_entropy"] = (
                    aux["sender_entropy"].float()[:, : self.message_len].mean().item()
                )
            self._log.write(json.dumps(row) + "\n")
        if self.step in self.record_steps:
            self.record(self.step, logs)

    def on_train_end(self):
        self._dyn.close()
        self._log.close()
        np.savez_compressed(
            os.path.join(self.out_dir, "protocol.npz"),
            steps=np.asarray(self.rec_steps),
            greedy_messages=np.stack(self.rec_greedy) if self.rec_greedy else np.zeros(0),
            mi=np.stack(self.rec_mi) if self.rec_mi else np.zeros(0),
            probe_states=self.probe_labels.cpu().numpy(),
            probe_split=self.probe_split,
        )
