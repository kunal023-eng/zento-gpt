# zento

A small GPT I built to actually understand how transformers work. Character
level, trained on tiny Shakespeare in PyTorch. Name is from 前途 (zento),
"the road ahead."

## Results

Trained the 4-layer, 4-head, 128-dim model on CPU in about 20 minutes.
Validation loss went from 4.31 (random guessing) to 1.63 at the end.
Sample output looks like this:

    KING RICHARD II:
    No, stir be heaven of so murself; grace to it neven
    That move hath been timpe the keep our bold,
    Than stopp'd with bed this did opence, tinks,

Names and structure are right, grammar is rough. That is expected for a
small model on 1MB of text.

## Example prompts and outputs

Different prompts, different continuations:

    $ uv run scripts/generate.py --prompt "ROMEO:" --tokens 150

    ROMEO:
    His quickens one.

    GREEN:
    Nor what beged a man face that beggd state seving.

    MENENIUS:
    What with he's to thy romb royal to safety,
    But this taking: will that close shings with eyes,

    $ uv run scripts/generate.py --prompt "JULIET:" --tokens 150

    JULIET:
    Whether it should out know, sir, know your shades,
    Will upon thee man. I say gave you:
    The hath scarce to so my wars never weak poff their werear
    Forted voices friendship aspieal thee fast.

    $ uv run scripts/generate.py --prompt "KING:" --tokens 80

    KING: the curse; Which thou to say, which a ginden.

    Tellmenge:
    He tribune nobth pois

    $ uv run scripts/generate.py --prompt "MARCIUS:" --tokens 100

    MARCIUS:
    Ah?

    Clown:
    We will one the elderit of my pightors.

    Fighty shall bear speak, but straights

The model learned that lines ending in `NAME:` are dialogue, so it continues
with new speaker tags and new lines. For non-character-name prompts it still
writes in the same style, which means it picked up the format of the training
data, not the content.

## Setup

    uv sync

Torch is pinned to CPU so it runs on any laptop without CUDA.

## Run

    uv run scripts/prepare_data.py
    uv run scripts/train_model.py
    uv run scripts/generate.py --prompt "ROMEO:" --tokens 300

The checkpoint goes to `experiment/gpt2.pt`. Not committed. To retrain,
just run `scripts/train_model.py` again (about 20 minutes on CPU).

## Structure

    src/zento_gpt/data.py     dataset + char tokenizer
    src/zento_gpt/model.py    model code (bigram, head, block, gpt)
    src/zento_gpt/train.py    training and eval loop
    scripts/                  runnable entry points
    tests/                    shape and mask checks
    data/                     dataset (not committed)
    experiment/               checkpoints (not committed)
    report/                   samples and plots

## Model

Small decoder-only transformer:

- 4 layers, 4 heads, 128 embedding dim
- context length 64
- token and position embeddings, causal self-attention, feedforward
- pre-layernorm, residual connections

## What I built in order

Started with a bigram model (lookup table, no context, val loss 2.49), then
a single attention head (val 2.33), then the full transformer block with
multi-head attention, feedforward, residuals, and layernorm (val 1.63).
The git history shows each step.

## Notes

- Character-level tokenizer, vocab is 65 for Shakespeare
- CPU only, no GPU on this laptop
- Trains in ~20 minutes

## Todo

- [ ] try top-k and temperature sampling
- [ ] loss curve plot in report/
- [ ] scale to 6 layers

## References

- Attention Is All You Need (Vaswani et al, 2017)
- Karpathy's "Let's build GPT" video, where I started

## License

MIT
