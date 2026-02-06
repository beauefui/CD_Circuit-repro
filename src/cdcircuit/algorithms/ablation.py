from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from cdcircuit.core.types import AblationSet, Node, TargetNodeDecompositionList


def reshape_separate_attention_heads(context_layer: Tensor, sa_module) -> Tensor:
    """Reshape [..., all_head_size] -> [..., n_heads, d_head]."""
    new_shape = context_layer.size()[:-1] + (
        sa_module.num_attention_heads,
        sa_module.attention_head_size,
    )
    return context_layer.view(new_shape)


def reshape_concatenate_attention_heads(context_layer: Tensor, sa_module) -> Tensor:
    """Reshape [..., n_heads, d_head] -> [..., all_head_size]."""
    new_shape = context_layer.size()[:-2] + (sa_module.all_head_size,)
    return context_layer.view(*new_shape)


def set_rel_at_source_nodes(
    rel: Tensor,
    irrel: Tensor,
    ablation_dict: Mapping[AblationSet, list[int]],
    layer_mean_acts: Tensor | None,
    layer_idx: int,
    sa_module,
    set_irrel_to_mean: bool,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Set selected source nodes to be fully relevant at this layer."""
    rel_heads = reshape_separate_attention_heads(rel, sa_module).clone()
    irrel_heads = reshape_separate_attention_heads(irrel, sa_module).clone()

    mean_heads = None
    if layer_mean_acts is not None:
        mean_heads = reshape_separate_attention_heads(layer_mean_acts, sa_module)
        if mean_heads.dim() == 3:
            mean_heads = mean_heads[None, :, :, :]

    for ablation, batch_indices in ablation_dict.items():
        for source_node in ablation:
            if source_node.layer_idx != layer_idx:
                continue

            sq = source_node.sequence_idx
            head = source_node.attn_head_idx
            total = rel_heads[batch_indices, sq, head, :] + irrel_heads[batch_indices, sq, head, :]

            if set_irrel_to_mean:
                if mean_heads is None:
                    raise ValueError("set_irrel_to_mean=True requires layer_mean_acts")
                mean_val = torch.as_tensor(mean_heads[:, sq, head, :], device=device, dtype=rel.dtype)
                rel_heads[batch_indices, sq, head, :] = total - mean_val
                irrel_heads[batch_indices, sq, head, :] = mean_val
            else:
                rel_heads[batch_indices, sq, head, :] = total
                irrel_heads[batch_indices, sq, head, :] = 0

    return (
        reshape_concatenate_attention_heads(rel_heads, sa_module),
        reshape_concatenate_attention_heads(irrel_heads, sa_module),
    )


def calculate_contributions(
    rel: Tensor,
    irrel: Tensor,
    ablation_dict: Mapping[AblationSet, list[int]],
    target_nodes: list[Node],
    level: int,
    sa_module,
) -> list[TargetNodeDecompositionList]:
    """Collect decomposition values at target nodes for each ablation set."""
    rel_heads = reshape_separate_attention_heads(rel, sa_module)
    irrel_heads = reshape_separate_attention_heads(irrel, sa_module)

    target_nodes_at_level = [node for node in target_nodes if node.layer_idx == level]
    out: list[TargetNodeDecompositionList] = []

    for ablation, batch_indices in ablation_dict.items():
        per_ablation = TargetNodeDecompositionList(ablation)
        for node in target_nodes_at_level:
            per_ablation.append(
                node,
                rel_heads[batch_indices, node.sequence_idx, node.attn_head_idx, :],
                irrel_heads[batch_indices, node.sequence_idx, node.attn_head_idx, :],
            )
        out.append(per_ablation)

    return out
