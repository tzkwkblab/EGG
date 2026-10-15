# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class DiscriminationLoss:
    def __call__(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        return self.discrimination_loss(receiver_output, labels)

    @staticmethod
    def discrimination_loss(receiver_output, labels):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}


class ReconstructionLoss:
    def __init__(self, n_attributes: int, n_values: int, batch_size: int):
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.batch_size = batch_size

    def __call__(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        return self.reconstruction_loss(
            receiver_output, labels, self.batch_size, self.n_attributes, self.n_values
        )

    @staticmethod
    def reconstruction_loss(
        receiver_output, labels, batch_size, n_attributes, n_values
    ):
        receiver_output = receiver_output.view(batch_size * n_attributes, n_values)
        receiver_guesses = receiver_output.argmax(dim=1)
        correct_samples = (
            (receiver_guesses == labels.view(-1))
            .view(batch_size, n_attributes)
            .detach()
        )
        acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()
        labels = labels.view(batch_size * n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        loss = loss.view(batch_size, -1).mean(dim=1)
        return loss, {"acc": acc}


class NTXentLoss:
    """NTXentLoss as originally described in https://arxiv.org/abs/2002.05709.

    This loss is used in self-supervised learning setups and requires the two views of the input datapoint
    to be returned distinctly by Sender and Receiver.
    Note that this loss considers in-batch negatives and and negatives samples are taken within each agent
    datapoints i.e. each non-target element in sender_input and in receiver_input is considered a negative sample.

    >>> x_i = torch.eye(128)
    >>> x_j = torch.eye(128)
    >>> loss_fn = NTXentLoss()
    >>> loss, aux = loss_fn(None, x_i, None, x_j, None, None)
    >>> aux["acc"].mean().item()
    1.0
    >>> aux["acc"].shape
    torch.Size([256])
    >>> x_i = torch.eye(256)
    >>> x_j = torch.eye(128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    Traceback (most recent call last):
        ...
    RuntimeError: sender_output and receiver_output must be of the same shape, found ... instead
    >>> _ = torch.manual_seed(111)
    >>> x_i = torch.rand(128, 128)
    >>> x_j = torch.rand(128, 128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    >>> aux['acc'].mean().item() * 100  # chance level with a batch size of 128, 1/128 * 100 = 0.78125
    0.78125
    """

    def __init__(
        self,
        temperature: float = 1.0,
        similarity: str = "cosine",
    ):
        self.temperature = temperature

        similarities = {"cosine", "dot"}
        assert (
            similarity.lower() in similarities
        ), f"Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    @staticmethod
    def ntxent_loss(
        sender_output: torch.Tensor,
        receiver_output: torch.Tensor,
        temperature: float = 1.0,
        similarity: str = "cosine",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if sender_output.shape != receiver_output.shape:
            raise RuntimeError(
                f"sender_output and receiver_output must be of the same shape, "
                f"found {sender_output.shape} and {receiver_output.shape} instead"
            )
        batch_size = sender_output.shape[0]

        input = torch.cat((sender_output, receiver_output), dim=0)

        if similarity == "cosine":
            similarity_f = torch.nn.CosineSimilarity(dim=2)
            similarity_matrix = (
                similarity_f(input.unsqueeze(1), input.unsqueeze(0)) / temperature
            )
        elif similarity == "dot":
            similarity_matrix = input @ input.t()

        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            batch_size * 2, 1
        )

        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool).fill_diagonal_(
            0
        )

        negative_samples = similarity_matrix[mask].reshape(batch_size * 2, -1)

        labels = torch.zeros(batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = F.cross_entropy(logits, labels, reduction="none") / 2

        acc = (torch.argmax(logits.detach(), dim=1) == labels).float().detach()
        return loss, {"acc": acc}

    def __call__(
        self,
        _sender_input,
        message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        return self.ntxent_loss(
            message,
            receiver_output,
            temperature=self.temperature,
            similarity=self.similarity,
        )


# ---------------------------------------------------------------------------
# Noise-robust losses and attribute-wise reconstruction
# ---------------------------------------------------------------------------

ROBUST_DEFAULTS = {
    "ce": {},
    "gce": {"kappa": 0.7},
    "sce": {"alpha": 0.2, "beta": 1.0, "A": -4.0},
}


def _log_target_prob(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """log-probability of the target class; logits [..., C], target [...] -> [...]"""
    return F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)


def _criterion_params(criterion: str, params: Optional[Dict[str, float]]) -> Dict:
    if criterion not in ROBUST_DEFAULTS:
        raise ValueError(f"Unknown criterion: {criterion}")
    merged = dict(ROBUST_DEFAULTS[criterion])
    if params:
        merged.update({k: v for k, v in params.items() if k in merged})
    return merged


def loss_from_log_prob(
    log_q: torch.Tensor, criterion: str = "ce", **params
) -> torch.Tensor:
    """Per-element loss as a function of the log-probability of the target class.

    ce:  -log q
    gce: (1 - q^kappa) / kappa                       (Zhang & Sabuncu, 2018)
    sce: alpha * (-log q) + beta * (-A) * (1 - q)    (Wang et al., 2019; log 0 clipped to A)

    >>> log_q = torch.tensor([0.5, 0.1]).log()
    >>> torch.allclose(loss_from_log_prob(log_q, "ce"), -log_q)
    True
    >>> torch.allclose(loss_from_log_prob(log_q, "gce", kappa=1.0), torch.tensor([0.5, 0.9]))
    True
    """
    p = _criterion_params(criterion, params)
    if criterion == "ce":
        return -log_q
    if criterion == "gce":
        return (1.0 - torch.exp(p["kappa"] * log_q)) / p["kappa"]
    q = log_q.exp()
    return p["alpha"] * (-log_q) + p["beta"] * (-p["A"]) * (1.0 - q)


def robust_weight(q_y: torch.Tensor, criterion: str = "ce", **params) -> torch.Tensor:
    """Gradient weight w(q_y) such that d loss / d logit_c = -w(q_y) (y_c - q_c).

    ce: 1, gce: q^kappa, sce: alpha - beta * A * q. The result is detached.

    >>> robust_weight(torch.tensor([0.1]), "gce", kappa=1.0)
    tensor([0.1000])
    """
    p = _criterion_params(criterion, params)
    q_y = q_y.detach()
    if criterion == "ce":
        return torch.ones_like(q_y)
    if criterion == "gce":
        return q_y.pow(p["kappa"])
    return p["alpha"] - p["beta"] * p["A"] * q_y


def weighted_ce_from_log_prob(
    log_q: torch.Tensor, criterion: str = "gce", mode: str = "w", **params
) -> torch.Tensor:
    """Surrogate loss sg[omega] * (-log q) whose gradient is a re-weighted CE gradient.

    mode "w":          omega = w(q)             (same gradient as the robust loss itself)
    mode "batch_mean": omega = mean of w(q)     (only the overall scale, shared by all elements)
    mode "relative":   omega = w(q) / mean w(q) (only the relative weights)
    The mean is taken over all elements of `log_q`.
    """
    w = robust_weight(log_q.exp(), criterion, **params)
    if mode == "w":
        omega = w
    elif mode == "batch_mean":
        omega = w.mean().expand_as(w)
    elif mode == "relative":
        omega = w / w.mean()
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return omega * (-log_q)


def gce_loss(logits: torch.Tensor, target: torch.Tensor, kappa: float = 0.7):
    """Generalized cross entropy; logits [N, C], target [N] -> [N]"""
    return loss_from_log_prob(_log_target_prob(logits, target), "gce", kappa=kappa)


def sce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 1.0,
    A: float = -4.0,
):
    """Symmetric cross entropy; logits [N, C], target [N] -> [N]"""
    return loss_from_log_prob(
        _log_target_prob(logits, target), "sce", alpha=alpha, beta=beta, A=A
    )


def weighted_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: str = "gce",
    mode: str = "w",
    **params,
):
    """Weighted CE surrogate; logits [N, C], target [N] -> [N]. See `weighted_ce_from_log_prob`."""
    return weighted_ce_from_log_prob(
        _log_target_prob(logits, target), criterion, mode, **params
    )


class AttributeReconstructionLoss:
    """Reconstruction loss for inputs described by several categorical attributes.

    Receiver outputs the concatenation of per-attribute logits, [B, sum(n_values)]. The number of
    values can differ between attributes. Returns a per-example loss [B] and the auxiliary dict
    with `acc` (all attributes correct, [B]), `acc_or` (per-attribute correctness, [B, n]) and
    `q_y` (probability of the correct value, detached, [B, n]).

    :param n_values: number of values of each attribute
    :param criterion: "ce", "gce" or "sce"
    :param criterion_params: overrides of the criterion's hyper-parameters (see ROBUST_DEFAULTS)
    :param weight_mode: if set ("w", "batch_mean", "relative"), use the weighted-CE surrogate
        of `criterion` instead of the criterion itself
    :param attr_reduction: "mean" or "sum" over attributes
    :param labels_from: "labels" (integer attribute values [B, n] given as game labels) or
        "sender_input" (one-hot concatenation given as Sender's input)

    >>> loss = AttributeReconstructionLoss([2, 3])
    >>> out = torch.tensor([[5.0, 0.0, 0.0, 5.0, 0.0]])
    >>> l, aux = loss(None, None, None, out, torch.tensor([[0, 1]]), None)
    >>> aux["acc"].item(), l.shape
    (1.0, torch.Size([1]))
    """

    def __init__(
        self,
        n_values: Sequence[int],
        criterion: str = "ce",
        criterion_params: Optional[Dict[str, float]] = None,
        weight_mode: Optional[str] = None,
        attr_reduction: str = "mean",
        labels_from: str = "labels",
    ):
        self.n_values = [int(v) for v in n_values]
        self.n_attributes = len(self.n_values)
        self.criterion = criterion
        self.criterion_params = _criterion_params(criterion, criterion_params)
        self.weight_mode = weight_mode
        assert attr_reduction in ("mean", "sum")
        assert labels_from in ("labels", "sender_input")
        self.attr_reduction = attr_reduction
        self.labels_from = labels_from
        self._uniform = len(set(self.n_values)) == 1

    def _split(self, flat: torch.Tensor):
        return torch.split(flat, self.n_values, dim=-1)

    def get_labels(self, sender_input, labels) -> torch.Tensor:
        if self.labels_from == "labels":
            return labels.long()
        return torch.stack([x.argmax(dim=-1) for x in self._split(sender_input)], dim=1)

    def log_target_prob(self, receiver_output: torch.Tensor, labels: torch.Tensor):
        """-> (log q_y [B, n], predictions [B, n])"""
        if self._uniform:
            logits = receiver_output.view(
                receiver_output.size(0), self.n_attributes, self.n_values[0]
            )
            return _log_target_prob(logits, labels), logits.argmax(dim=-1)
        log_q, pred = [], []
        for i, logits in enumerate(self._split(receiver_output)):
            log_q.append(_log_target_prob(logits, labels[:, i]))
            pred.append(logits.argmax(dim=-1))
        return torch.stack(log_q, dim=1), torch.stack(pred, dim=1)

    def elementwise(self, log_q: torch.Tensor) -> torch.Tensor:
        if self.weight_mode is not None:
            return weighted_ce_from_log_prob(
                log_q, self.criterion, self.weight_mode, **self.criterion_params
            )
        return loss_from_log_prob(log_q, self.criterion, **self.criterion_params)

    def reduce(self, per_attr: torch.Tensor) -> torch.Tensor:
        if self.attr_reduction == "mean":
            return per_attr.mean(dim=-1)
        return per_attr.sum(dim=-1)

    def __call__(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input=None,
    ):
        labels = self.get_labels(sender_input, labels)
        log_q, pred = self.log_target_prob(receiver_output, labels)
        loss = self.reduce(self.elementwise(log_q))

        correct = (pred == labels).detach()
        acc = (correct.sum(dim=1) == self.n_attributes).float()
        aux = {"acc": acc, "acc_or": correct.float(), "q_y": log_q.detach().exp()}
        return loss, aux
