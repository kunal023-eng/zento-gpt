import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zento_gpt.data import download, prepare


def main():
    root = Path(__file__).resolve().parent.parent
    raw = root / "data" / "raw" / "shakespeare.txt"
    out = root / "data" / "processed"
    raw.parent.mkdir(parents=True, exist_ok=True)
    download(raw)
    prepare(raw, out)


if __name__ == "__main__":
    main()
