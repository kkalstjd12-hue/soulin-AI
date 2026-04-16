import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from sklearn.metrics import classification_report, f1_score
from preprocess import main as load_data, ID2LABEL, LABEL2ID


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CFG = {
    "model_name": "klue/roberta-large",
    "max_length": 128,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3,
    "seed": 42,
    "save_dir": "./checkpoints",
    "num_labels": 6,
}


class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            loss    = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


def train():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("데이터 전처리 중...")
    train_df, val_df = load_data()

    print(f"\n모델 로드: {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels=CFG["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_dataset = EmotionDataset(train_df, tokenizer, CFG["max_length"])
    val_dataset   = EmotionDataset(val_df,   tokenizer, CFG["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2)

    optimizer = AdamW(
        model.parameters(),
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )

    total_steps  = len(train_loader) * CFG["num_epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = torch.tensor(
        1.0 / class_counts / (1.0 / class_counts).sum(),
        dtype=torch.float,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    save_dir = Path(CFG["save_dir"])
    save_dir.mkdir(exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    print("\n=== 학습 시작 ===")

    for epoch in range(1, CFG["num_epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_f1, val_preds, val_labels = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(save_dir / "best_model")
            tokenizer.save_pretrained(save_dir / "best_model")
            print(f"  ✓ Best model 저장 (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping patience: {patience_counter}/{CFG['early_stopping_patience']}")
            if patience_counter >= CFG["early_stopping_patience"]:
                print("\nEarly stopping 발동!")
                break

        print()

    print(f"\n=== 학습 완료 | Best Val Macro F1: {best_f1:.4f} ===\n")
    print("최종 Classification Report:")
    label_names = [ID2LABEL[i] for i in range(CFG["num_labels"])]
    print(classification_report(val_labels, val_preds, target_names=label_names))


if __name__ == "__main__":
    train()
