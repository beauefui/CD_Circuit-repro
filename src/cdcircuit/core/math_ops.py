from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def assert_invariant(rel: Tensor, irrel: Tensor, total_ref: Tensor | None = None, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    if total_ref is None:
        return
    if not torch.allclose(rel + irrel, total_ref, atol=atol, rtol=rtol):
        diff = (rel + irrel - total_ref).abs().max().item()
        raise AssertionError(f"Decomposition invariant violated: max_abs_diff={diff:.6e}")


def normalize_rel_irrel(rel: Tensor, irrel: Tensor) -> tuple[Tensor, Tensor]:
    """Numerically stabilize decomposition when rel and irrel have opposite signs.

    Returns new tensors and preserves rel + irrel exactly elementwise.
    """
    total = rel + irrel

    sign_conflict = (rel * irrel) < 0
    rel_dominant = sign_conflict & (rel.abs() >= irrel.abs())
    irrel_dominant = sign_conflict & (~rel_dominant)

    rel_out = rel.clone()
    irrel_out = irrel.clone()

    rel_out[rel_dominant] = total[rel_dominant]
    irrel_out[rel_dominant] = 0

    rel_out[irrel_dominant] = 0
    irrel_out[irrel_dominant] = total[irrel_dominant]

    return rel_out, irrel_out


def prop_linear_core(rel: Tensor, irrel: Tensor, W: Tensor, b: Tensor, tol: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Propagate decomposition through affine map xW + b.

    Bias is allocated proportionally to |relW| and |irrelW|.
    """
    rel_t = torch.matmul(rel, W)
    irrel_t = torch.matmul(irrel, W)

    bias_expanded = b.expand_as(rel_t)
    total_weight = rel_t.abs() + irrel_t.abs() + tol

    rel_bias = bias_expanded * (rel_t.abs() / total_weight)
    irrel_bias = bias_expanded * (irrel_t.abs() / total_weight)

    return rel_t + rel_bias, irrel_t + irrel_bias


def prop_linear(rel: Tensor, irrel: Tensor, linear_module) -> tuple[Tensor, Tensor]:
    return prop_linear_core(rel, irrel, linear_module.weight.T, linear_module.bias)


def prop_activation(rel: Tensor, irrel: Tensor, act_fn: Callable[[Tensor], Tensor]) -> tuple[Tensor, Tensor]:
    irrel_act = act_fn(irrel)
    rel_act = act_fn(rel + irrel) - irrel_act
    return rel_act, irrel_act


def prop_layer_norm(rel: Tensor, irrel: Tensor, ln_module, tol: float = 1e-8) -> tuple[Tensor, Tensor]:
    total = rel + irrel

    rel_mean = rel.mean(dim=-1, keepdim=True)
    irrel_mean = irrel.mean(dim=-1, keepdim=True)
    var = total.var(dim=-1, unbiased=False, keepdim=True)

    rel_weight = rel.abs()
    irrel_weight = irrel.abs()
    total_weight = rel_weight + irrel_weight + tol

    rel_t = ((rel - rel_mean) / torch.sqrt(var + ln_module.eps)) * ln_module.weight
    irrel_t = ((irrel - irrel_mean) / torch.sqrt(var + ln_module.eps)) * ln_module.weight

    rel_bias = ln_module.bias * (rel_weight / total_weight)
    irrel_bias = ln_module.bias * (irrel_weight / total_weight)

    return rel_t + rel_bias, irrel_t + irrel_bias
