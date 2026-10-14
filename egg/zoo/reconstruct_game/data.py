# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch


def index_to_state(index: np.ndarray, n_values: Sequence[int]) -> np.ndarray:
    """Mixed-radix decoding of state indices; [N] -> [N, n_attributes].

    >>> index_to_state(np.array([0, 1, 5]), [2, 3]).tolist()
    [[0, 0], [0, 1], [1, 2]]
    """
    index = np.asarray(index, dtype=np.int64).copy()
    out = np.zeros((len(index), len(n_values)), dtype=np.int64)
    for i in reversed(range(len(n_values))):
        out[:, i] = index % n_values[i]
        index //= n_values[i]
    return out


def build_state_space(
    n_values: Sequence[int], n_train: int, n_test: int, split_seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draws disjoint train / test sets of states (integer attribute values, [N, n_attributes]).

    If the state space has at most n_train + n_test states, all of them are used (the test set
    takes n_test of them, the train set the rest); otherwise n_train + n_test distinct states are
    sampled uniformly.

    >>> train, test = build_state_space([3, 3], n_train=6, n_test=3, split_seed=0)
    >>> train.shape, test.shape, len({tuple(s) for s in torch.cat([train, test]).tolist()})
    (torch.Size([6, 2]), torch.Size([3, 2]), 9)
    """
    total = int(np.prod([int(v) for v in n_values], dtype=object))
    rng = np.random.RandomState(split_seed)
    wanted = n_train + n_test
    if total <= wanted:
        assert total > n_test, "the state space is not larger than the test set"
        index = rng.permutation(total)
    elif total <= 50_000_000:
        index = rng.choice(total, size=wanted, replace=False)
    else:  # too large to permute: rejection sampling of distinct indices
        chosen = set()
        while len(chosen) < wanted:
            chosen.update(rng.randint(0, total, size=wanted - len(chosen)).tolist())
        index = rng.permutation(np.fromiter(chosen, dtype=np.int64))
    states = torch.from_numpy(index_to_state(index, n_values))
    return states[n_test:], states[:n_test]


def one_hot_states(states: torch.Tensor, n_values: Sequence[int]) -> torch.Tensor:
    """Concatenation of per-attribute one-hot vectors; [N, n] -> [N, sum(n_values)].

    >>> one_hot_states(torch.tensor([[1, 0]]), [2, 3]).tolist()
    [[0.0, 1.0, 1.0, 0.0, 0.0]]
    """
    parts = [
        torch.nn.functional.one_hot(states[:, i], int(v)).float()
        for i, v in enumerate(n_values)
    ]
    return torch.cat(parts, dim=1)


def make_sender_inputs(
    states: torch.Tensor,
    n_values: Sequence[int],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sender's input for each state. By default the one-hot concatenation; alternatively
    pre-computed `features` (one row per state) or a deterministic `transform` of the states.
    The ground-truth attribute values are always kept separately as labels.
    """
    if features is not None:
        assert len(features) == len(states)
        return features.float()
    if transform is not None:
        return transform(states).float()
    return one_hot_states(states, n_values)
