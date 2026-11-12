# zento

A small GPT I'm building to actually understand how transformers work.
Character level, trained on tiny Shakespeare in PyTorch.

Name is from 前途 (zento), "the road ahead."

## Setup

    uv sync

Torch is pinned to CPU so it runs on any laptop without CUDA.

## Run

    uv run scripts/prepare_data.py
    uv run scripts/train_model.py
    uv run scripts/generate.py

## Structure

    src/zento_gpt/    model and training code
    scripts/          entry points you actually run
    data/             dataset (not committed)
    experiment/       checkpoints (not committed)
    report/           samples and plots
    tests/

## Model

Small decoder-only transformer. 4 layers, 4 heads, 128 dim, context 128.
Details in the code.

## Notes

- CPU training, about 30 min for the full run
- Character level, so vocab is small (65 for Shakespeare)

## Todo

- [ ] top-k and temperature sampling
- [ ] loss curve plot
- [ ] try the 6 layer config

## References

- Attention Is All You Need (Vaswani et al, 2017)
- Karpathy's "Let's build GPT" video, which is where I started

## License

MIT
