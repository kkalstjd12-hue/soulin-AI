import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CONFIG = {
    "processed_dir": "./processed_data",
    "output_dir": "./checkpoints",

    "max_length": 128,          
    "batch_size": 32,           
    "num_epochs": 5,
    "learning_rate": 2e-5,      
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,        
    "threshold": 0.5,           
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

class EthicsDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record["text"]
        labels = record["labels"]  # List[float], length=NUM_LABELS

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [max_length]
            "labels":         torch.tensor(labels, dtype=torch.float), # [NUM_LABELS]
        }

def evaluate(model, dataloader, pos_weight, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float).to(device)
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)  # [N, NUM_LABELS]
    all_labels = np.vstack(all_labels)  # [N, NUM_LABELS]

    # sigmoid → 이진 예측
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= threshold).astype(int)

    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    per_class_f1 = f1_score(all_labels, preds, average=None, zero_division=0)

    avg_loss = total_loss / len(dataloader)
    return avg_loss, macro_f1, micro_f1, per_class_f1

def train():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    print("\n=== 데이터 로드 ===")
    with open(f"{cfg['processed_dir']}/train_records.pkl", "rb") as f:
        train_records = pickle.load(f)
    with open(f"{cfg['processed_dir']}/val_records.pkl", "rb") as f:
        val_records = pickle.load(f)
    pos_weight = np.load(f"{cfg['processed_dir']}/pos_weight.npy")

    with open(f"{cfg['processed_dir']}/meta.json", "r") as f:
        meta = json.load(f)
    LABELS     = meta["labels"]
    NUM_LABELS = meta["num_labels"]

    print(f"Train: {len(train_records):,} | Val: {len(val_records):,}")
    print(f"Labels: {LABELS}")

    print(f"\n=== 모델 로드: {cfg['model_name']} ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model.to(device)

    train_dataset = EthicsDataset(train_records, tokenizer, cfg["max_length"])
    val_dataset   = EthicsDataset(val_records,   tokenizer, cfg["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg["learning_rate"])

    total_steps   = len(train_loader) * cfg["num_epochs"]
    warmup_steps  = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float).to(device)
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    best_macro_f1 = 0.0
    best_epoch    = 0

    print(f"\n=== 학습 시작 (총 {cfg['num_epochs']} epoch) ===")
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                # [batch, NUM_LABELS]
            loss   = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if (step + 1) % 100 == 0:
                avg_loss = total_train_loss / (step + 1)
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, macro_f1, micro_f1, per_class_f1 = evaluate(
            model, val_loader, pos_weight, device, cfg["threshold"]
        )

        print(f"\n[Epoch {epoch}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Macro F1:   {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
        print(f"  Per-class F1:")
        for i, label in enumerate(LABELS):
            print(f"    {label:<20}: {per_class_f1[i]:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch    = epoch
            model.save_pretrained(f"{cfg['output_dir']}/best_model")
            tokenizer.save_pretrained(f"{cfg['output_dir']}/best_model")
            print(f"  ✅ Best model saved! (Macro F1: {best_macro_f1:.4f})")

    print(f"\n=== 학습 완료 ===")
    print(f"  Best Epoch: {best_epoch} | Best Macro F1: {best_macro_f1:.4f}")
    print(f"  저장 경로: {cfg['output_dir']}/best_model")

def predict(texts, model_dir="./checkpoints/best_model", threshold=0.5):
    """
    texts: List[str] — 예측할 텍스트 리스트
    반환: List[List[str]] — 각 텍스트의 레이블 리스트
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    with open("./processed_data/meta.json", "r") as f:
        meta = json.load(f)
    LABELS = meta["labels"]

    encoding = tokenizer(
        texts,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)

    results = []
    for i, text in enumerate(texts):
        predicted_labels = [LABELS[j] for j in range(len(LABELS)) if preds[i][j] == 1]
        results.append({
            "text": text,
            "predicted_labels": predicted_labels if predicted_labels else ["IMMORAL_NONE"],
            "probabilities": {LABELS[j]: float(probs[i][j]) for j in range(len(LABELS))},
        })
    return results


if __name__ == "__main__":
    train()
