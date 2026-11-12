import torch


def get_batch(data, batch_size, block_size, device):
    n = len(data)
    starts = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def estimate_loss(model, train_data, val_data, batch_size, block_size, device, n=50):
    model.eval()
    out = {}
    with torch.no_grad():
        for name, data in [("train", train_data), ("val", val_data)]:
            losses = [model(*get_batch(data, batch_size, block_size, device))[1].item()
                      for _ in range(n)]
            out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(model, train_data, val_data, device, steps, batch_size, block_size,
          lr, eval_every, save_path=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        xb, yb = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(xb, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            e = estimate_loss(model, train_data, val_data, batch_size, block_size, device)
            print(f"step {step:5d}  train {e['train']:.3f}  val {e['val']:.3f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print("saved", save_path)

    return model
