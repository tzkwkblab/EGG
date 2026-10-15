# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from egg.core.losses import (
    AttributeReconstructionLoss,
    gce_loss,
    robust_weight,
    sce_loss,
    weighted_ce_loss,
)
from egg.zoo.compo_vs_generalization.train import DiffLoss


def _data(n=64, c=7, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, c, generator=g, dtype=dtype) * 2
    target = torch.randint(0, c, (n,), generator=g)
    return logits, target


def test_gce_matches_definition():
    logits, target = _data()
    q = F.softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
    for kappa in (0.3, 0.7, 1.0):
        assert torch.allclose(gce_loss(logits, target, kappa), (1 - q**kappa) / kappa)


def test_gce_tends_to_ce():
    logits, target = _data()
    ce = F.cross_entropy(logits, target, reduction="none")
    assert torch.allclose(gce_loss(logits, target, 1e-6), ce, rtol=1e-4, atol=1e-5)


def test_sce_matches_definition():
    logits, target = _data()
    alpha, beta, A = 0.2, 1.0, -4.0
    p = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(target, logits.size(1)).double()
    # reverse CE with log(0) clipped to A: -sum_c p_c * log(y_c)
    log_y = torch.where(one_hot > 0, torch.zeros_like(p), torch.full_like(p, A))
    rce = -(p * log_y).sum(dim=1)
    ce = F.cross_entropy(logits, target, reduction="none")
    assert torch.allclose(sce_loss(logits, target, alpha, beta, A), alpha * ce + beta * rce)


def test_robust_weight_is_the_logit_gradient_weight():
    # d loss / d logit_c = -w(q_y) (y_c - q_c)
    for criterion, fn in (("gce", gce_loss), ("sce", sce_loss)):
        logits, target = _data()
        logits.requires_grad_(True)
        fn(logits, target).sum().backward()
        p = F.softmax(logits.detach(), dim=-1)
        q = p.gather(1, target[:, None]).squeeze(1)
        one_hot = F.one_hot(target, logits.size(1)).double()
        expected = -robust_weight(q, criterion)[:, None] * (one_hot - p)
        assert torch.allclose(logits.grad, expected, rtol=1e-8, atol=1e-10)


def test_weighted_ce_has_the_gradient_of_the_robust_loss():
    for criterion, fn in (("gce", gce_loss), ("sce", sce_loss)):
        logits, target = _data()
        a = logits.clone().requires_grad_(True)
        b = logits.clone().requires_grad_(True)
        fn(a, target).mean().backward()
        weighted_ce_loss(b, target, criterion, "w").mean().backward()
        rel = (a.grad - b.grad).abs().max() / a.grad.abs().max()
        assert rel < 1e-8


def test_weight_decomposition():
    logits, target = _data()
    ce = F.cross_entropy(logits, target, reduction="none")
    for criterion in ("gce", "sce"):
        w = weighted_ce_loss(logits, target, criterion, "w") / ce
        scale = weighted_ce_loss(logits, target, criterion, "batch_mean") / ce
        rel = weighted_ce_loss(logits, target, criterion, "relative") / ce
        assert torch.allclose(scale, scale[0].expand_as(scale))
        assert torch.allclose(rel.mean(), torch.ones((), dtype=rel.dtype))
        assert torch.allclose(scale * rel, w)


def test_attribute_loss_ce_equals_diffloss():
    n_att, n_val, bsz = 3, 5, 16
    g = torch.Generator().manual_seed(1)
    labels = torch.randint(0, n_val, (bsz, n_att), generator=g)
    sender_input = F.one_hot(labels, n_val).float().view(bsz, -1)
    output = torch.randn(bsz, n_att * n_val, generator=g)

    ref_loss, ref_aux = DiffLoss(n_att, n_val)(sender_input, None, None, output, None, None)
    for labels_from, lab in (("labels", labels), ("sender_input", None)):
        loss, aux = AttributeReconstructionLoss([n_val] * n_att, labels_from=labels_from)(
            sender_input, None, None, output, lab, None
        )
        assert torch.allclose(loss, ref_loss, atol=1e-6)
        assert torch.equal(aux["acc"], ref_aux["acc"])
        assert torch.equal(aux["acc_or"], ref_aux["acc_or"])


def test_attribute_loss_heterogeneous_values():
    n_values = [2, 3, 6]
    g = torch.Generator().manual_seed(2)
    labels = torch.stack([torch.randint(0, v, (8,), generator=g) for v in n_values], dim=1)
    output = torch.randn(8, sum(n_values), generator=g)
    parts = torch.split(output, n_values, dim=1)
    for criterion, fn in (("ce", F.cross_entropy), ("gce", gce_loss), ("sce", sce_loss)):
        kwargs = {"reduction": "none"} if criterion == "ce" else {}
        expected = torch.stack(
            [fn(parts[i], labels[:, i], **kwargs) for i in range(3)], dim=1
        )
        loss, aux = AttributeReconstructionLoss(n_values, criterion)(
            None, None, None, output, labels, None
        )
        assert torch.allclose(loss, expected.mean(dim=1), atol=1e-6)
        assert aux["q_y"].shape == (8, 3) and aux["acc_or"].shape == (8, 3)
        loss_sum, _ = AttributeReconstructionLoss(n_values, criterion, attr_reduction="sum")(
            None, None, None, output, labels, None
        )
        assert torch.allclose(loss_sum, expected.sum(dim=1), atol=1e-6)
