import torch

from cdcircuit.core.masks import get_extended_attention_mask


class DummyModel(torch.nn.Module):
    def __init__(self, is_decoder: bool):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1))

        class Cfg:
            pass

        self.config = Cfg()
        self.config.is_decoder = is_decoder


def test_encoder_mask_shape():
    model = DummyModel(is_decoder=False)
    mask = torch.tensor([[1, 1, 0, 0]])
    out = get_extended_attention_mask(mask, input_shape=(1, 4), model=model)
    assert out.shape == (1, 1, 1, 4)


def test_decoder_mask_shape():
    model = DummyModel(is_decoder=True)
    mask = torch.tensor([[1, 1, 1, 0]])
    out = get_extended_attention_mask(mask, input_shape=(1, 4), model=model)
    assert out.shape == (1, 1, 4, 4)


def test_3d_mask_shape():
    model = DummyModel(is_decoder=False)
    mask = torch.ones(2, 4, 4)
    out = get_extended_attention_mask(mask, input_shape=(2, 4), model=model)
    assert out.shape == (2, 1, 4, 4)
