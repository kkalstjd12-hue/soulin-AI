import json
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

DATA_ROOT = "./147.텍스트 윤리검증 데이터/01.데이터"
TRAIN_DIR = f"{DATA_ROOT}/1.Training/라벨링데이터/aihub/TL1_aihub"
VAL_DIR   = f"{DATA_ROOT}/2.Validation/라벨링데이터/aihub/talksets-train-6"
OUTPUT_DIR = "./processed_data"

LABELS = [
    "IMMORAL_NONE",
    "CENSURE",
    "HATE",
    "DISCRIMINATION",
    "SEXUAL",
    "VIOLENCE",
    "ABUSE",
    "CRIME",
]
LABEL2ID   = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

TRAIN_FILES = [
    f"{TRAIN_DIR}/talksets-train-1/talksets-train-1_aihub.json",
    f"{TRAIN_DIR}/talksets-train-2/talksets-train-2.json",
    f"{TRAIN_DIR}/talksets-train-3/talksets-train-3.json",
    f"{TRAIN_DIR}/talksets-train-4/talksets-train-4.json",
    f"{TRAIN_DIR}/talksets-train-5/talksets-train-5.json",
]
VAL_FILE = f"{VAL_DIR}/talksets-train-6.json"

import re

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_and_flatten(filepaths):
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    records = []
    for fp in filepaths:
        with open(fp, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        for conv in conversations:
            for sent in conv.get("sentences", []):
                text = clean_text(sent.get("text", ""))
                if not text:
                    continue
                types = sent.get("types", [])
                label_vector = [0.0] * NUM_LABELS
                for t in types:
                    if t in LABEL2ID:
                        label_vector[LABEL2ID[t]] = 1.0
                records.append({"text": text, "labels": label_vector})
        print(f"  로드 완료: {os.path.basename(fp)} → {len(records):,} 문장 누적")
    return records

def compute_pos_weight(records):
    n = len(records)
    pos_counts = np.array([sum(r["labels"][i] for r in records) for i in range(NUM_LABELS)])
    neg_counts = n - pos_counts
    return neg_counts / (pos_counts + 1e-8)

def print_stats(records, name="Dataset"):
    print(f"\n=== {name} 통계 ===")
    print(f"총 문장 수: {len(records):,}")
    label_counts = Counter()
    for r in records:
        for i, v in enumerate(r["labels"]):
            if v > 0:
                label_counts[LABELS[i]] += 1
    for label in LABELS:
        cnt = label_counts[label]
        print(f"  {label:<20}: {cnt:>7,}개 ({cnt/len(records)*100:.1f}%)")
    multi = sum(1 for r in records if sum(r["labels"]) > 1)
    print(f"  멀티레이블 문장: {multi:,} ({multi/len(records)*100:.1f}%)")

if __name__ == "__main__":
    print("=== Train 데이터 로딩 ===")
    train_records = load_and_flatten(TRAIN_FILES)
    print("\n=== Validation 데이터 로딩 ===")
    val_records = load_and_flatten(VAL_FILE)

    print_stats(train_records, "Train")
    print_stats(val_records,   "Validation")

    pos_weight = compute_pos_weight(train_records)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/train_records.pkl", "wb") as f:
        pickle.dump(train_records, f)
    with open(f"{OUTPUT_DIR}/val_records.pkl", "wb") as f:
        pickle.dump(val_records, f)
    np.save(f"{OUTPUT_DIR}/pos_weight.npy", pos_weight)
    with open(f"{OUTPUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "label2id": LABEL2ID, "num_labels": NUM_LABELS,
                   "train_size": len(train_records), "val_size": len(val_records)}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 전처리 완료! → {OUTPUT_DIR}/")
