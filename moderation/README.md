# 텍스트 윤리검증 AI 모더레이션

한국어 텍스트의 유해성을 분석하는 Multi-label 분류 모델입니다.

## 🔗 링크
- **HuggingFace 모델**: [MINSEONG12/moderation](https://huggingface.co/MINSEONG12/moderation)
- **API 서버 (HuggingFace Space)**: [MINSEONG12-moderation.hf.space](https://MINSEONG12-moderation.hf.space)
- **API 문서 (Swagger UI)**: [MINSEONG12-moderation.hf.space/docs](https://MINSEONG12-moderation.hf.space/docs)

---

## 📌 프로젝트 개요

온라인 커뮤니티의 유해 게시글 및 댓글을 자동으로 탐지하기 위한 AI 모더레이션 시스템입니다.  
단순 이진 분류(정상/유해)를 넘어, **8가지 유해 유형을 동시에 감지**하는 Multi-label 분류 방식을 채택했습니다.

---

## 🏷️ 감지 레이블

| 레이블 | 설명 |
|--------|------|
| `IMMORAL_NONE` | 정상 (유해하지 않음) |
| `CENSURE` | 비난 / 비하 |
| `HATE` | 혐오 발언 |
| `DISCRIMINATION` | 차별 발언 |
| `SEXUAL` | 성적 발언 |
| `VIOLENCE` | 폭력적 발언 |
| `ABUSE` | 욕설 |
| `CRIME` | 범죄 관련 발언 |

---

## 🧠 모델 정보

| 항목 | 내용 |
|------|------|
| 베이스 모델 | [KLUE/BERT-base](https://huggingface.co/klue/bert-base) |
| 학습 방식 | Multi-label Fine-tuning |
| 데이터 | AIHub 텍스트 윤리검증 데이터셋 (약 35만+ 문장) |
| 평가 지표 | Macro F1 |
| 클래스 불균형 처리 | BCEWithLogitsLoss + pos_weight |
| 문장 분리 | kss (Korean Sentence Splitter) |
| 배포 | HuggingFace Space (Docker + FastAPI) |

---

## 📁 파일 구조

```
moderation/
├── README.md           # 프로젝트 설명
├── preprocess.py       # 데이터 전처리 (JSON → Multi-label 벡터)
├── train.py            # KLUE/BERT 파인튜닝 학습
├── predict.py          # 추론 (단문 / 장문 모두 지원)
└── requirements.txt    # 의존성
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

---

## 🚀 사용법

### 1. 데이터 전처리
```bash
python preprocess.py
```
AIHub 데이터 JSON → 문장 단위 Multi-label 벡터로 변환 후 `processed_data/` 에 저장

### 2. 모델 학습
```bash
python train.py
```
KLUE/BERT-base 파인튜닝, 베스트 모델은 `checkpoints/best_model/` 에 저장

### 3. 추론
```bash
python predict.py --text "분석할 텍스트"

# 임계값 조정 (낮을수록 민감)
python predict.py --text "분석할 텍스트" --threshold 0.4
```

### 출력 예시
```
=== 전체 판정: ['DISCRIMINATION', 'VIOLENCE'] ===

✅ [1] 오늘 날씨 좋다.
🚨 [2] 저 여자는 왜 저러는지 모르겠네.
      └─ 감지: ['DISCRIMINATION', 'CENSURE']
🚨 [3] 때려버리고 싶다.
      └─ 감지: ['VIOLENCE']
```

---

## 🌐 API 호출

HuggingFace Space에 배포된 REST API를 바로 사용할 수 있습니다.

```python
import requests

response = requests.post(
    "https://MINSEONG12-moderation.hf.space/moderate",
    json={"text": "분석할 텍스트"}
)
print(response.json())
```

```json
{
  "overall_labels": ["DISCRIMINATION"],
  "is_harmful": true,
  "sentence_results": [
    {"sentence": "오늘 날씨 좋다.", "labels": ["IMMORAL_NONE"], "is_harmful": false},
    {"sentence": "저 여자는 왜 저러는지.", "labels": ["DISCRIMINATION"], "is_harmful": true}
  ]
}
```

---

## 📊 데이터

- **출처**: [AIHub 텍스트 윤리검증 데이터](https://aihub.or.kr)
- **Train**: talksets-train-1 ~ 5 (약 35만 문장)
- **Validation**: talksets-train-6
- 데이터 파일은 AIHub 사용 약관에 따라 레포에 포함하지 않습니다.
