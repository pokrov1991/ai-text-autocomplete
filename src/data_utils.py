import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "raw_dataset.csv"
PROCESSED_PATH = DATA_DIR / "dataset_processed.csv"

def _clean_text(text):
    text = text.lower()  # к нижнему регистру
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # убираем ссылки
    text = re.sub(r"@\w+", "", text)  # убираем упоминания
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # только буквы и цифры
    text = re.sub(r"\s+", " ", text).strip()  # убираем дублирующиеся пробелы
    return text

def create_dataset_processed():
    # Построчное чтение (без CSV-парсера, чтобы не споткнуться о запятые в тексте)
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()] # убираем пустые строки

    cleaned = [_clean_text(line) for line in lines]

    df = pd.DataFrame({
        "text_raw": lines, # исходная строка
        "text": cleaned, # очищенный текст
    })

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[OK] Создан очищенный датасет {PROCESSED_PATH}, строк: {len(df)}")
    return len(df)

def split_dataset():
    # Берёт dataset_processed.csv и делит его на train/val/test (80/10/10)
    df = pd.read_csv(PROCESSED_PATH)

    # train 80%, temp 20%
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    # из temp берём 10% вал, 10% тест
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Cохраняем
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"[OK] Созданы тренировочная, валидационная, тестовая выборки: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return len(train_df), len(val_df), len(test_df)

if __name__ == "__main__":
    create_dataset_processed()
    split_dataset()
