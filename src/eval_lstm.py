# Замер метрик lstm модели
import torch
import evaluate


@torch.no_grad() # не нужно считать градиенты (ускоряем вычисления)
def _generate_autocomplete(model, prefix_ids: torch.Tensor, need_len: int) -> torch.Tensor:
    """
    Метод пошагово дописывает токены к префиксу
    Прогоняем весь префикс через модель, берём argmax последнего шага
    """
    model.eval()
    seq = prefix_ids.clone()
    for _ in range(need_len):
        logits = model(seq.unsqueeze(0))             # [1, T, V]
        next_tok = logits[:, -1, :].argmax(dim=-1)   # [1] выбираем токен с максимальной вероятностью
        seq = torch.cat([seq, next_tok], dim=0)      # [T] + [1] -> [T+1] добавляем его в конец
    return seq


@torch.no_grad() # не нужно считать градиенты (ускоряем вычисления)
def eval_rouge(model, data_loader, tokenizer, pad_id: int, max_batches: int = None):
    """
    Сценарий 3/4 → 1/4: модель получает первые 3/4 токенов и дописывает последние 1/4
    Считаем ROUGE (rouge1, rouge2)
    """
    rouge = evaluate.load("rouge")
    predictions, references = [], []

    device = next(model.parameters()).device

    for bi, batch in enumerate(data_loader):
        if max_batches is not None and bi >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        mask = (input_ids != pad_id).long()
        lens = mask.sum(dim=1)

        for i in range(input_ids.size(0)):
            L = int(lens[i].item())
            if L < 4:
                continue

            full = input_ids[i, :L]
            split = max(1, int(0.75 * L))
            prefix = full[:split]
            target = full[split:]
            need = len(target)
            if need == 0:
                continue

            pred_full = _generate_autocomplete(model, prefix, need_len=need)
            pred_tail = pred_full[-need:]

            pred_text = tokenizer.decode(pred_tail.tolist(), skip_special_tokens=True)
            ref_text = tokenizer.decode(target.tolist(), skip_special_tokens=True)

            if pred_text.strip() and ref_text.strip():
                predictions.append(pred_text)
                references.append(ref_text)

    results = rouge.compute(predictions=predictions, references=references)
    return results


@torch.no_grad() # не нужно считать градиенты (ускоряем вычисления)
def autocomplete_examples(model, data_loader, tokenizer, pad_id: int, num_examples: int = 5):
    """
    Показываем примеры автодополнений (FULL / PREFIX / TARGET / PRED).
    """
    device = next(model.parameters()).device
    shown = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        lens = (input_ids != pad_id).long().sum(dim=1)
        for i in range(input_ids.size(0)):
            if shown >= num_examples:
                return
            L = int(lens[i].item())
            if L < 4:
                continue

            full = input_ids[i, :L]
            split = max(1, int(0.75 * L))
            prefix, target = full[:split], full[split:]
            need = len(target)

            pred_full = _generate_autocomplete(model, prefix, need_len=need)
            pred_tail = pred_full[-need:]

            print("—" * 60)
            print("FULL:   ", tokenizer.decode(full.tolist(), skip_special_tokens=True))
            print("PREFIX: ", tokenizer.decode(prefix.tolist(), skip_special_tokens=True))
            print("TARGET: ", tokenizer.decode(target.tolist(), skip_special_tokens=True))
            print("PRED:   ", tokenizer.decode(pred_tail.tolist(), skip_special_tokens=True))
            shown += 1
