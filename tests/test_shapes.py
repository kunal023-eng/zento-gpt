import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zento_gpt.model import Bigram, Head, MultiHead, Block, GPT


def test_bigram_shape():
    m = Bigram(65)
    logits, _ = m(torch.randint(0, 65, (2, 8)))
    assert logits.shape == (2, 8, 65)


def test_head_shape():
    h = Head(64, 16, 8)
    assert h(torch.randn(2, 8, 64)).shape == (2, 8, 16)


def test_multihead_shape():
    m = MultiHead(64, 4, 8)
    assert m(torch.randn(2, 8, 64)).shape == (2, 8, 64)


def test_block_shape():
    b = Block(64, 4, 8)
    assert b(torch.randn(2, 8, 64)).shape == (2, 8, 64)


def test_gpt_shape():
    m = GPT(65, 64, 4, 2, 16)
    logits, _ = m(torch.randint(0, 65, (2, 8)))
    assert logits.shape == (2, 8, 65)


def test_mask_is_causal():
    h = Head(8, 4, 4)
    assert h.mask[0, 1] == 0
    assert h.mask[1, 0] == 1
    assert h.mask[3, 3] == 1


def test_no_future_leak():
    # position 0 must not attend to position 1
    h = Head(8, 4, 4)
    m = h.mask
    for i in range(4):
        for j in range(i + 1, 4):
            assert m[i, j] == 0
