import json
import pandas as pd
from pathlib import Path

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()
TRAIN_JSON = BASE_DIR / "Training_221115_add/라벨링데이터/감성대화말뭉치(최종데이터)_Training.json"
VAL_JSON   = BASE_DIR / "Validation_221115_add/라벨링데이터/감성대화말뭉치(최종데이터)_Validation.json"
TRAIN_XLSX = BASE_DIR / "Training_221115_add/원천데이터/감성대화말뭉치(최종데이터)_Training..xlsx"
VAL_XLSX   = BASE_DIR / "Validation_221115_add/원천데이터/감성대화말뭉치(최종데이터)_Validation.xlsx"

LABEL2ID = {
    "분노": 0,
    "불안": 1,
    "당황": 2,
    "슬픔": 3,
    "상처": 4,
    "기쁨": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def build_text(content: dict) -> str:
    turns = []
    for key in ["HS01", "HS02", "HS03"]:
        val = content.get(key, "").strip()
        if val:
            turns.append(val)
    return " [SEP] ".join(turns)


def load_xlsx_labels(xlsx_path: Path) -> dict:
    df = pd.read_excel(xlsx_path, header=0)
    df.columns = [str(c).strip() if str(c) != "None" else "idx" for c in df.columns]
    return df[["감정_대분류"]].copy()


def process_json_with_xlsx(json_path: Path, xlsx_df: pd.DataFrame) -> pd.DataFrame:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for i, item in enumerate(data):
        content = item["talk"]["content"]
        text = build_text(content)

        label_str = xlsx_df.iloc[i]["감정_대분류"]
        label_id = LABEL2ID.get(label_str, -1)

        records.append({
            "text": text,
            "label": label_id,
            "label_str": label_str,
        })

    df = pd.DataFrame(records)

    df = df[df["label"] != -1].reset_index(drop=True)
    return df


def process_val_json(json_path: Path, val_xlsx_path: Path) -> pd.DataFrame:

    val_xlsx_df = load_xlsx_labels(val_xlsx_path)
    return process_json_with_xlsx(json_path, val_xlsx_df)


def main():
    print("=== 데이터 전처리 시작 ===\n")

    print("[1/2] Training 데이터 로드 중...")
    train_xlsx_df = load_xlsx_labels(TRAIN_XLSX)
    train_df = process_json_with_xlsx(TRAIN_JSON, train_xlsx_df)
    print(f"  Train 샘플 수: {len(train_df)}")
    print(f"  레이블 분포:\n{train_df['label_str'].value_counts().to_string()}\n")

    print("[2/2] Validation 데이터 로드 중...")
    val_df = process_val_json(VAL_JSON, VAL_XLSX)
    print(f"  Val 샘플 수: {len(val_df)}")
    print(f"  레이블 분포:\n{val_df['label_str'].value_counts().to_string()}\n")

    train_df.to_csv(BASE_DIR / "train_processed.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(BASE_DIR / "val_processed.csv", index=False, encoding="utf-8-sig")
    print("저장 완료: train_processed.csv / val_processed.csv")

    print("\n=== 샘플 확인 ===")
    print(train_df[["text", "label_str"]].head(3).to_string())

    return train_df, val_df


if __name__ == "__main__":
    main()
