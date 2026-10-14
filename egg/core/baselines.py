# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch


class Baseline(ABC):
    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        """Update internal state according to the observed loss
        loss (torch.Tensor): batch of losses
        """
        pass

    @abstractmethod
    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        """Return baseline for the loss
        Args:
            loss (torch.Tensor): batch of losses be baselined
        """
        pass


class NoBaseline(Baseline):
    """Baseline that does nothing (constant zero baseline)"""

    def __init__(self):
        super().__init__()

    def update(self, loss: torch.Tensor) -> None:
        pass

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, device=loss.device)


class MeanBaseline(Baseline):
    """Running mean baseline; all loss batches have equal importance/weight,
    hence it is better if they are equally-sized.
    """

    def __init__(self):
        super().__init__()

        self.mean_baseline = torch.zeros(1, requires_grad=False)
        self.n_points = 0.0

    def update(self, loss: torch.Tensor) -> None:
        self.n_points += 1
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)

        self.mean_baseline += (
            loss.detach().mean().item() - self.mean_baseline
        ) / self.n_points

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)
        return self.mean_baseline


class BuiltInBaseline(Baseline):
    """Built-in baseline; for any row in the batch, the mean of all other rows serves as a control variate.
    To use BuiltInBaseline, rows in the batch must be independent. Most likely BuiltInBaseline
    would work poorly for small batch sizes.
    """

    def __init__(self):
        super().__init__()

    def update(self, _: torch.Tensor) -> None:
        pass

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if len(loss.size()) == 0 or loss.size(0) <= 1:
            return loss
        bsz = loss.size(0)

        loss_detached = loss.detach()
        mean = loss_detached.mean()

        baseline = (mean * bsz - loss_detached) / (bsz - 1.0)

        return baseline


class MLPCritic(torch.nn.Module):
    """State-value function (critic) that does not share parameters with Sender.

    Predicts the per-example loss from Sender's input; use it as `value_fn` of
    `SenderReceiverRnnReinforce` to get an actor-critic style baseline.

    >>> critic = MLPCritic(n_inputs=6, n_hidden=4)
    >>> critic(torch.zeros(3, 6)).shape
    torch.Size([3])
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_layers: int = 1):
        super().__init__()
        layers, d = [], n_inputs
        for _ in range(n_layers):
            layers += [torch.nn.Linear(d, n_hidden), torch.nn.ReLU()]
            d = n_hidden
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, sender_input, aux_input=None):
        return self.net(sender_input).squeeze(-1)


class SharedFeatureCritic(torch.nn.Module):
    """State-value function (critic) computed from the features of Sender's input encoder.

    `encoder` is the very module Sender uses to encode its input (called as `encoder(x, aux_input)`),
    so its parameters are shared. With `detach_input=True` the critic's loss does not propagate
    into the encoder.
    """

    def __init__(
        self, encoder: torch.nn.Module, n_hidden: int, detach_input: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(n_hidden, 1)
        self.detach_input = detach_input

    def forward(self, sender_input, aux_input=None):
        features = self.encoder(sender_input, aux_input)
        if self.detach_input:
            features = features.detach()
        return self.head(features).squeeze(-1)
