import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zento_gpt.data import Tokenizer
from zento_gpt.model import Bigram
from zento_gpt.train import train


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ids = torch.from_numpy(np.load(data_dir / "train.npy")).long()
    val_ids = torch.from_numpy(np.load(data_dir / "val.npy")).long()
    tok = Tokenizer.load(data_dir / "tokenizer.pkl")

    model = Bigram(tok.vocab_size)
    print(f"device {device}  params {sum(p.numel() for p in model.parameters())}")

    train(
        model, train_ids, val_ids, device,
        steps=3000, batch_size=32, block_size=64, lr=3e-3, eval_every=500,
        save_path=str(root / "experiment" / "bigram.pt"),
    )

    print("\nsample:")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, 200)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
