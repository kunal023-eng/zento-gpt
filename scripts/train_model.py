import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zento_gpt.data import Tokenizer
from zento_gpt.model import GPT
from zento_gpt.train import train


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ids = torch.from_numpy(np.load(data_dir / "train.npy")).long()
    val_ids = torch.from_numpy(np.load(data_dir / "val.npy")).long()
    tok = Tokenizer.load(data_dir / "tokenizer.pkl")

    model = GPT(
        vocab_size=tok.vocab_size,
        n_embd=128, n_head=4, n_layer=4, block_size=64, dropout=0.1,
    )
    print(f"device {device}  params {sum(p.numel() for p in model.parameters())}")

    train(
        model, train_ids, val_ids, device,
        steps=5000, batch_size=32, block_size=64, lr=1e-3, eval_every=500,
        save_path=str(root / "experiment" / "gpt2.pt"),
    )

    print("\nsample:")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, 500)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
