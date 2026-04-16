# 한국어 텍스트 윤리검증 AI 모더레이션

KLUE-BERT 기반 한국어 텍스트 유해성 분류 모델입니다.  
AI Hub 텍스트 윤리검증 데이터셋을 활용해 8개 유해 유형을 Multi-label로 분류합니다.

---

## 모델 정보

| 항목 | 내용 |
|------|------|
| 베이스 모델 | klue/bert-base |
| 허깅페이스 | https://huggingface.co/MINSEONG12/moderation |
| API 서버 | https://MINSEONG12-moderation.hf.space |
| 학습 데이터 | AI Hub 텍스트 윤리검증 데이터셋 |
| Train / Val | talksets-train-1~5 / talksets-train-6 |
| 분류 방식 | Multi-label Classification (8개 레이블) |
| 감지 레이블 | IMMORAL_NONE, CENSURE, HATE, DISCRIMINATION, SEXUAL, VIOLENCE, ABUSE, CRIME |

---

## 성능

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| IMMORAL_NONE | xx | xx | xx |
| CENSURE | xx | xx | xx |
| HATE | xx | xx | xx |
| DISCRIMINATION | xx | xx | xx |
| SEXUAL | xx | xx | xx |
| VIOLENCE | xx | xx | xx |
| ABUSE | xx | xx | xx |
| CRIME | xx | xx | xx |
| **Macro F1** | | | **xx** |

---

## 파일 구조

```
├── README.md
├── requirements.txt
├── preprocess.py       # 데이터 전처리
├── train.py            # 모델 학습
└── predict.py          # 추론
```

---

## 실행 방법

**설치**
```bash
pip install -r requirements.txt
```

**전처리**
```bash
python preprocess.py
```

**학습**
```bash
python train.py
```

**추론**
```bash
python predict.py --text "분석할 텍스트"
```

**추론 (임계값 조정)**
```bash
python predict.py --text "분석할 텍스트" --threshold 0.4
```

---

## 학습 설정

| 항목 | 값 |
|------|----|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Max Epochs | 5 |
| Max Length | 128 |
| Class Imbalance | BCEWithLogitsLoss + pos_weight |
| Warmup | Linear warmup (10%) |

---

## 추론 방식

입력 텍스트를 `kss`로 문장 단위로 분리한 후 각 문장에 대해 독립적으로 Multi-label 분류를 수행합니다.

```
입력 텍스트
    ↓ kss 문장 분리
[문장1, 문장2, 문장3, ...]
    ↓ 각 문장 개별 추론 (sigmoid + threshold)
[레이블1, 레이블2, 레이블3, ...]
    ↓ 결과 집계
전체 판정 + 문장별 유해 위치 반환
```

---

## API

HuggingFace Space에 FastAPI로 배포되어 있습니다.

```python
import requests

response = requests.post(
    "https://MINSEONG12-moderation.hf.space/moderate",
    json={"text": "분석할 텍스트"}
)
print(response.json())
```

---

## 향후 개선 계획

- max_length 512로 재학습
- 하이퍼파라미터 튜닝 (learning rate, batch size)
- 임계값 레이블별 최적화 실험

---

## 데이터셋 출처

AI Hub 텍스트 윤리검증 데이터  
https://www.aihub.or.kr
