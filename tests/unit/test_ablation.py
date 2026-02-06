import torch

from cdcircuit.algorithms.ablation import (
    calculate_contributions,
    reshape_concatenate_attention_heads,
    reshape_separate_attention_heads,
    set_rel_at_source_nodes,
)
from cdcircuit.core.types import Node


class DummySAModule:
    num_attention_heads = 2
    attention_head_size = 3
    all_head_size = 6


def test_head_reshape_roundtrip():
    sa = DummySAModule()
    x = torch.randn(4, 5, sa.all_head_size)
    sep = reshape_separate_attention_heads(x, sa)
    cat = reshape_concatenate_attention_heads(sep, sa)
    assert sep.shape == (4, 5, 2, 3)
    assert torch.allclose(x, cat)


def test_set_rel_at_source_nodes_zero_irrel():
    sa = DummySAModule()
    rel = torch.zeros(2, 4, 6)
    irrel = torch.ones(2, 4, 6)

    ablation = (Node(0, 1, 0),)
    ablation_dict = {ablation: [0, 1]}

    rel_o, irrel_o = set_rel_at_source_nodes(
        rel,
        irrel,
        ablation_dict,
        layer_mean_acts=None,
        layer_idx=0,
        sa_module=sa,
        set_irrel_to_mean=False,
        device="cpu",
    )

    rel_h = reshape_separate_attention_heads(rel_o, sa)
    irrel_h = reshape_separate_attention_heads(irrel_o, sa)

    assert torch.allclose(rel_h[:, 1, 0, :], torch.ones(2, 3))
    assert torch.allclose(irrel_h[:, 1, 0, :], torch.zeros(2, 3))


def test_set_rel_at_source_nodes_with_mean():
    sa = DummySAModule()
    rel = torch.zeros(1, 2, 6)
    irrel = torch.ones(1, 2, 6)
    mean = torch.full((2, 6), 0.25)

    ablation = (Node(3, 0, 1),)
    ablation_dict = {ablation: [0]}

    rel_o, irrel_o = set_rel_at_source_nodes(
        rel,
        irrel,
        ablation_dict,
        layer_mean_acts=mean,
        layer_idx=3,
        sa_module=sa,
        set_irrel_to_mean=True,
        device="cpu",
    )

    rel_h = reshape_separate_attention_heads(rel_o, sa)
    irrel_h = reshape_separate_attention_heads(irrel_o, sa)

    assert torch.allclose(irrel_h[0, 0, 1, :], torch.full((3,), 0.25))
    assert torch.allclose(rel_h[0, 0, 1, :], torch.full((3,), 0.75))


def test_calculate_contributions_picks_level_targets():
    sa = DummySAModule()
    rel = torch.arange(2 * 3 * 6, dtype=torch.float32).view(2, 3, 6)
    irrel = rel + 1000

    ablation_a = (Node(0, 0, 0),)
    ablation_b = (Node(1, 1, 1),)
    ablation_dict = {ablation_a: [0], ablation_b: [1]}

    targets = [Node(2, 1, 1), Node(0, 0, 0), Node(2, 2, 0)]
    out = calculate_contributions(rel, irrel, ablation_dict, targets, level=2, sa_module=sa)

    assert len(out) == 2
    assert len(out[0].target_nodes) == 2
    assert out[0].target_nodes[0] == Node(2, 1, 1)
    assert out[0].rels[0].shape == (1, 3)
