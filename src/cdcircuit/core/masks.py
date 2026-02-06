from __future__ import annotations

import warnings

import torch
from torch import Tensor
from transformers.modeling_utils import ModuleUtilsMixin


def _is_decoder_model(model) -> bool:
    if hasattr(model, "config") and getattr(model.config, "is_decoder", False):
        return True

    # TransformerLens HookedTransformer compatibility without importing transformer_lens here.
    if model.__class__.__name__ == "HookedTransformer":
        return True

    return False


def get_extended_attention_mask(attention_mask: Tensor, input_shape: tuple[int, ...], model, device=None) -> Tensor:
    """Mirror HF mask expansion used by old implementation.

    Output is additive attention bias: 0 for keep, -inf for masked positions.
    """
    dtype = next(model.parameters()).dtype
    is_decoder = _is_decoder_model(model)

    if not (attention_mask.dim() == 2 and is_decoder) and device is not None:
        warnings.warn(
            "The `device` argument is deprecated and only kept for compatibility.",
            FutureWarning,
            stacklevel=2,
        )

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if is_decoder:
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask
