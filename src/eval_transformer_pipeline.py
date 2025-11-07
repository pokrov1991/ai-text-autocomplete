# Используем distilgpt2 для автодополнения и оценки ROUGE
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate
import pandas as pd
import torch

@torch.no_grad() # не нужно считать градиенты (ускоряем вычисления)
def eval_transformer_rouge(data_path="data/val.csv", text_col="text", max_samples=200):
    """
    Оценивает качество distilgpt2 на валидационном датасете.
    Делим текст на 3/4 (prefix) и 1/4 (target), генерируем продолжение.
    Возвращает словарь с метриками ROUGE.
    """

    # 1. Модель и токенизатор
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Настройки для предотвращения ворнингов
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

    # 2. Загружаем тексты
    df = pd.read_csv(data_path)
    texts = df[text_col].dropna().tolist()[:max_samples]

    def split_text(text):
        words = text.split()
        if len(words) < 8:
            return None
        cut = int(len(words) * 0.75)
        prefix = " ".join(words[:cut])
        target = " ".join(words[cut:])
        return prefix, target

    preds, refs = [], []
    examples = []

    print(f"[RUN] distilgpt2 на {len(texts)} примерах ({'CUDA' if device == 0 else 'CPU'}).")

    # 3. Генерация и сбор предсказаний
    for text in texts:
        parts = split_text(text)
        if not parts:
            continue
        prefix, target = parts
        target_wc = len(target.split())
        out = generator(
            prefix,
            max_new_tokens=max(1, target_wc),
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            truncation=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        gen_text = out[0]["generated_text"]
        continuation = gen_text[len(prefix):].strip()
        preds.append(continuation)
        refs.append(target)

        if len(examples) < 5:
            examples.append((text, prefix, target, continuation))

    # 4. Метрики ROUGE
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs)

    print("[OK] Метрики ROUGE (distilgpt2):")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    print("[OK] Пример автозаполнений:")
    for i, (text, prefix, target, pred) in enumerate(examples, start=1):
        print("—" * 60)
        print("FULL:  ", text)
        print("PREFIX:", prefix)
        print("TARGET:", target)
        print("PRED  :", pred)

    return scores
