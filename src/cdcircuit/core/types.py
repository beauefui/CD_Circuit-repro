from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch


class Node(NamedTuple):
    layer_idx: int
    sequence_idx: int
    attn_head_idx: int


AblationSet = tuple[Node, ...]


class OutputDecomposition(NamedTuple):
    ablation_set: AblationSet
    rel: torch.Tensor
    irrel: torch.Tensor


@dataclass
class TargetNodeDecompositionList:
    ablation_set: AblationSet
    target_nodes: list[Node] = field(default_factory=list)
    rels: list[torch.Tensor] = field(default_factory=list)
    irrels: list[torch.Tensor] = field(default_factory=list)

    def append(self, target_node: Node, rel: torch.Tensor, irrel: torch.Tensor) -> None:
        self.target_nodes.append(target_node)
        self.rels.append(rel)
        self.irrels.append(irrel)

    def __add__(self, other: "TargetNodeDecompositionList") -> "TargetNodeDecompositionList":
        if self.ablation_set != other.ablation_set:
            raise ValueError("Cannot merge decomposition lists with different ablation sets")
        merged = TargetNodeDecompositionList(self.ablation_set)
        merged.target_nodes = self.target_nodes + other.target_nodes
        merged.rels = self.rels + other.rels
        merged.irrels = self.irrels + other.irrels
        return merged
