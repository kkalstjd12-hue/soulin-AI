import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import kss

MODEL_DIR = "MINSEONG12/moderation"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "IMMORAL_NONE",    # 정상
    "CENSURE",         # 비난/비하
    "HATE",            # 혐오
    "DISCRIMINATION",  # 차별
    "SEXUAL",          # 성적
    "VIOLENCE",        # 폭력
    "ABUSE",           # 욕설
    "CRIME",           # 범죄
]

def load_model():
    print(f"모델 로드 중: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print("모델 로드 완료\n")
    return tokenizer, model

def predict_long_text(text, tokenizer, model, threshold=0.5):
    """
    텍스트를 문장 단위로 분리 후 각 문장 유해성 분석

    Args:
        text      : 분석할 텍스트 (단문 / 장문 모두 가능)
        tokenizer : 로드된 토크나이저
        model     : 로드된 모델
        threshold : 유해 판정 임계값 (기본 0.5)

    Returns:
        dict: overall_labels, is_harmful, sentence_results
    """
    sentences = kss.split_sentences(text)

    enc = tokenizer(
        sentences,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)

    all_labels = set()
    sentence_results = []

    for i, sent in enumerate(sentences):
        predicted = [LABELS[j] for j in range(len(LABELS)) if preds[i][j] == 1]
        if not predicted:
            predicted = ["IMMORAL_NONE"]
        all_labels.update(predicted)
        sentence_results.append({
            "sentence"  : sent,
            "labels"    : predicted,
            "is_harmful": predicted != ["IMMORAL_NONE"],
        })

    harmful_labels = list(all_labels - {"IMMORAL_NONE"})

    return {
        "overall_labels"  : harmful_labels or ["IMMORAL_NONE"],
        "is_harmful"      : bool(harmful_labels),
        "sentence_results": sentence_results,
    }


def print_result(result):
    """결과 출력"""
    print(f"=== 전체 판정: {result['overall_labels']} ===\n")
    for i, r in enumerate(result["sentence_results"]):
        flag = "🚨" if r["is_harmful"] else "✅"
        print(f"{flag} [{i+1}] {r['sentence']}")
        if r["is_harmful"]:
            print(f"      └─ 감지: {r['labels']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="텍스트 유해성 분석")
    parser.add_argument("--text",      type=str,   required=True, help="분석할 텍스트")
    parser.add_argument("--threshold", type=float, default=0.5,   help="유해 판정 임계값 (기본: 0.5)")
    args = parser.parse_args()

    tokenizer, model = load_model()
    result = predict_long_text(args.text, tokenizer, model, args.threshold)
    print_result(result)
