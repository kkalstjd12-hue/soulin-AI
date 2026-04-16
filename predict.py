import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import LABEL2ID, ID2LABEL, build_text


CHECKPOINT = "./checkpoints/best_model"
MAX_LENGTH = 128


def load_model(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model     = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def predict_single(text: str, tokenizer, model, device) -> dict:
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs      = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id    = int(probs.argmax())
    pred_label = ID2LABEL[pred_id]

    return {
        "prediction": pred_label,
        "confidence": float(probs[pred_id]),
        "scores": {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
    }


def predict_from_turns(hs01: str, hs02: str = "", hs03: str = "",
                       tokenizer=None, model=None, device=None) -> dict:
    content = {"HS01": hs01, "HS02": hs02, "HS03": hs03}
    text = build_text(content)
    return predict_single(text, tokenizer, model, device)


def predict_batch(csv_path: str, tokenizer, model, device) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    predictions = []
    confidences = []

    for text in df["text"]:
        result = predict_single(str(text), tokenizer, model, device)
        predictions.append(result["prediction"])
        confidences.append(result["confidence"])

    df["prediction"]  = predictions
    df["confidence"]  = confidences
    return df


def interactive_mode(tokenizer, model, device):
    print("=== 감정 분류 대화형 테스트 ===")
    print("종료: 'q' 입력\n")

    while True:
        hs01 = input("사람 발화 1 (HS01): ").strip()
        if hs01.lower() == "q":
            break
        hs02 = input("사람 발화 2 (HS02, 없으면 엔터): ").strip()
        hs03 = input("사람 발화 3 (HS03, 없으면 엔터): ").strip()

        result = predict_from_turns(hs01, hs02, hs03, tokenizer, model, device)
        print(f"\n예측 감정: {result['prediction']} (confidence: {result['confidence']:.2%})")
        print("전체 점수:")
        for label, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"  {label:4s}: {score:.4f} {bar}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",       type=str, default=None, help="배치 추론용 CSV 파일 경로")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT, help="모델 체크포인트 경로")
    args = parser.parse_args()

    print(f"모델 로드 중: {args.checkpoint}")
    tokenizer, model, device = load_model(args.checkpoint)

    if args.file:
        print(f"배치 추론: {args.file}")
        result_df = predict_batch(args.file, tokenizer, model, device)
        out_path = args.file.replace(".csv", "_predicted.csv")
        result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"결과 저장: {out_path}")
        print(result_df[["text", "prediction", "confidence"]].head(10))
    else:
        interactive_mode(tokenizer, model, device)


if __name__ == "__main__":
    main()
