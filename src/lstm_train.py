# Код обучения модели
import torch
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=100):
    """
    Один проход обучения по train_loader.
    Возвращает средний train loss.
    """
    model.train()
    running = 0.0

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)  # [B, T, V]
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += loss.item()
        if step % log_every == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f}")

    return running / max(1, step)


@torch.no_grad() # не нужно считать градиенты (ускоряем вычисления)
def validate(model, loader, criterion, device):
    """
    Валидация без градиентов.
    Возвращает средний val loss.
    """
    model.eval()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total += loss.item()
    return total / max(1, len(loader))
