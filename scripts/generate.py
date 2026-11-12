import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zento_gpt.data import Tokenizer
from zento_gpt.model import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--ckpt", default="experiment/gpt2.pt")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer.load(root / "data" / "processed" / "tokenizer.pkl")
    model = GPT(tok.vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=64)
    model.load_state_dict(torch.load(root / args.ckpt, map_location=device))
    model.to(device)
    model.eval()

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, args.tokens)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
