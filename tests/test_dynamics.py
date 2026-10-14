# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import numpy as np
import torch

from egg.core.dynamics import default_record_steps, preserved_rng
from egg.core.language_analysis import (
    Disent,
    compute_bosdis,
    compute_posdis,
    compute_topsim_from_pairs,
    gap_mi_first_second,
    gap_mi_first_second_fast,
    mutual_info_matrix,
    sample_pairs,
)


def test_record_steps():
    steps = default_record_steps(20000)
    assert steps[:11] == [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    assert steps[11] == 1500 and steps[-1] == 20000
    assert len(steps) == 49
    assert default_record_steps(300) == [0, 1, 2, 5, 10, 20, 50, 100, 200, 300]


def test_compositional_language_is_maximal():
    attributes = np.array(list(itertools.product(range(4), repeat=3)))
    # one position per attribute, with a separate set of symbols for each position
    # (with shared symbols, the edit distance can be smaller than the Hamming distance)
    messages = attributes + 1 + 4 * np.arange(3)
    assert abs(compute_topsim_from_pairs(attributes, messages) - 1.0) < 1e-9
    assert abs(compute_posdis(attributes, messages) - 1.0) < 1e-9
    mi = mutual_info_matrix(attributes, messages)
    assert np.allclose(mi, 2.0 * np.eye(3))


def test_constant_messages_give_nan_topsim():
    attributes = np.array(list(itertools.product(range(3), repeat=2)))
    assert np.isnan(compute_topsim_from_pairs(attributes, np.ones_like(attributes)))


def test_fast_gap_matches_original():
    g = torch.Generator().manual_seed(0)
    attributes = torch.randint(0, 4, (200, 3), generator=g)
    messages = torch.randint(1, 5, (200, 4), generator=g)
    messages[:, 0] = attributes[:, 1] + 1
    messages[:, 3] = 1  # constant position
    assert abs(
        gap_mi_first_second(attributes, messages)
        - gap_mi_first_second_fast(attributes, messages)
    ) < 1e-5
    assert abs(
        Disent.posdis(attributes, messages) - compute_posdis(attributes, messages)
    ) < 1e-5
    assert abs(
        Disent.bosdis(attributes, messages, vocab_size=5)
        - compute_bosdis(attributes, messages, vocab_size=5)
    ) < 1e-5


def test_sample_pairs():
    assert len(sample_pairs(10, None)) == 45
    pairs = sample_pairs(100, 500, seed=1)
    assert pairs.shape == (500, 2) and (pairs[:, 0] != pairs[:, 1]).all()
    assert np.array_equal(pairs, sample_pairs(100, 500, seed=1))


def test_preserved_rng():
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    with preserved_rng(123):
        torch.rand(10)
    assert torch.equal(torch.rand(3), expected)
