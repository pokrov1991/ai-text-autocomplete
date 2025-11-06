
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import torch

def _read_texts(csv_path: Path):
    # Берем колонку 'text', иначе пробуем 'text_raw'
    df = pd.read_csv(csv_path)
    if "text" in df.columns:
        return df["text"].astype(str).tolist()
    elif "text_raw" in df.columns:
        return df["text_raw"].astype(str).tolist()
    else:
        # Первая колонка
        first_col = df.columns[0]
        return df[first_col].astype(str).tolist()


def _pick_eos_id(tokenizer):
    # Если нет токена конца последовательности EOS, то берем токен-разделитель предложений SEP
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id
    if getattr(tokenizer, "sep_token_id", None) is not None:
        return tokenizer.sep_token_id
    # Если вообще ничего нет — не будем добавлять EOS
    return None


class NextTokenDataset(Dataset):
    """
    Для каждого текста:
      - ids = tokenizer.encode(text, add_special_tokens=False)
      - добавляем EOS в конец последовательности (при условии add_eos === True,)
      - извлекаем ВСЕ окна длиной (seq_len+1): X = chunk[:-1], Y = chunk[1:]  (сдвиг на 1 токен)
    """
    def __init__(
        self,
        texts,
        tokenizer,
        seq_len = 32,
        add_eos = True,
        max_length_per_text = 2048,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.eos_id = _pick_eos_id(tokenizer) if add_eos else None

        self.samples: List[Tuple[List[int], List[int]]] = []

        for text in texts:
            # Токенизация без спец-токенов (EOS добавляем при условии)
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length_per_text,
            )
            if self.eos_id is not None:
                ids = ids + [self.eos_id]

            need = self.seq_len + 1
            if len(ids) < need:
                continue

            # Обрабатывая каждый токен, таргет смещаем на 1 токен вправо относительно исходной последовательности
            for i in range(0, len(ids) - need + 1):
                chunk = ids[i:i + need]       # длина = seq_len+1
                x = chunk[:-1]                # входы
                y = chunk[1:]                 # таргеты (сдвиг вправо на 1)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
        }


def _collate_shifted(batch, pad_id):
    """
    Паддинг справа:
      - input_ids паддим pad_id (из tokenizer.pad_token_id)
      - labels паддим -100, чтобы игнорировать в CrossEntropyLoss
    """
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad(vec: torch.Tensor, value: int):
        out = torch.full((max_len,), value, dtype=vec.dtype)
        out[: len(vec)] = vec
        return out

    input_ids = torch.stack([pad(item["input_ids"], pad_id) for item in batch])
    labels    = torch.stack([pad(item["labels"],   -100)   for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def create_loaders(
    data_dir,
    tokenizer_name = "bert-base-multilingual-cased",
    seq_len = 32,
    batch_size = 64,
    add_eos = True,
    max_length_per_text = 2048,
    shuffle_train = True,
):
    """
    Создаем лоадеры:
      - грузим тексты из data/train.csv, data/val.csv, data/test.csv
      - токенизируем через tokenizer (без спец-токенов)
      - формируем X/Y со сдвигом на 1 на уровне каждого текста
      - отдаём три DataLoader
    """
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv"
    val_csv   = data_dir / "val.csv"
    test_csv  = data_dir / "test.csv"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        # Если у токенизатора нет пад-токена, попробуем назначить
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.eos_token or "[PAD]"

    train_texts = _read_texts(train_csv)
    val_texts   = _read_texts(val_csv)
    test_texts  = _read_texts(test_csv)

    train_ds = NextTokenDataset(train_texts, tokenizer, seq_len=seq_len, add_eos=add_eos, max_length_per_text=max_length_per_text)
    val_ds   = NextTokenDataset(val_texts,   tokenizer, seq_len=seq_len, add_eos=add_eos, max_length_per_text=max_length_per_text)
    test_ds  = NextTokenDataset(test_texts,  tokenizer, seq_len=seq_len, add_eos=add_eos, max_length_per_text=max_length_per_text)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    def collate_fn(batch):
        return _collate_shifted(batch, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"[OK] Созданы Dataset и DataLoader для обучения модели.")
    return (train_ds, val_ds, test_ds), (train_loader, val_loader, test_loader), tokenizer
