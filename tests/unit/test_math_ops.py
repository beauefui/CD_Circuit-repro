import torch

from cdcircuit.core.math_ops import (
    assert_invariant,
    normalize_rel_irrel,
    prop_activation,
    prop_layer_norm,
    prop_linear_core,
)


def test_normalize_rel_irrel_preserves_sum():
    rel = torch.tensor([[1.0, -2.0, 3.0]])
    irrel = torch.tensor([[-2.0, 1.0, 4.0]])

    rel_n, irrel_n = normalize_rel_irrel(rel, irrel)
    assert torch.allclose(rel + irrel, rel_n + irrel_n)


def test_prop_linear_core_invariant():
    rel = torch.randn(2, 3)
    irrel = torch.randn(2, 3)
    W = torch.randn(3, 4)
    b = torch.randn(4)

    rel_o, irrel_o = prop_linear_core(rel, irrel, W, b)
    ref = torch.matmul(rel + irrel, W) + b
    assert_invariant(rel_o, irrel_o, ref, atol=1e-5, rtol=1e-4)


def test_prop_activation_invariant_relu():
    rel = torch.randn(3, 5)
    irrel = torch.randn(3, 5)
    rel_o, irrel_o = prop_activation(rel, irrel, torch.relu)
    ref = torch.relu(rel + irrel)
    assert_invariant(rel_o, irrel_o, ref, atol=1e-6, rtol=1e-5)


def test_prop_layer_norm_output_shape():
    ln = torch.nn.LayerNorm(8)
    rel = torch.randn(2, 4, 8)
    irrel = torch.randn(2, 4, 8)

    rel_o, irrel_o = prop_layer_norm(rel, irrel, ln)
    assert rel_o.shape == rel.shape
    assert irrel_o.shape == irrel.shape
