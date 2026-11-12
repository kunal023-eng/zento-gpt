import os
import pickle
import urllib.request
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download(path):
    if os.path.exists(path):
        print("already have", path)
        return
    print("downloading...")
    urllib.request.urlretrieve(URL, path)


class Tokenizer:
    def __init__(self, chars):
        self.chars = sorted(chars)
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(set(text))

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def prepare(raw_path, out_dir, split=0.9):
    with open(raw_path, encoding="utf-8") as f:
        text = f.read()

    tok = Tokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.uint16)
    n = int(len(ids) * split)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train.npy"), ids[:n])
    np.save(os.path.join(out_dir, "val.npy"), ids[n:])
    tok.save(os.path.join(out_dir, "tokenizer.pkl"))

    print(f"chars {len(text)}, vocab {tok.vocab_size}, train {n}, val {len(ids) - n}")
