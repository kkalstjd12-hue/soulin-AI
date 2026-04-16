# 한국어 감정 분류 AI

KLUE-RoBERTa 기반 한국어 텍스트 감정 분류 모델입니다.  
AI Hub 감성대화 말뭉치 데이터셋을 활용해 6개 감정 클래스를 분류합니다.

---

## 모델 정보

| 항목 | 내용 |
|------|------|
| 베이스 모델 | klue/roberta-large |
| 허깅페이스 | https://huggingface.co/MINSEONG12/emotion_classifier |
| 학습 데이터 | AI Hub 감성대화 말뭉치 |
| Train / Val | 51,628 / 6,640 |
| 분류 클래스 | 분노, 불안, 당황, 슬픔, 상처, 기쁨 |

---

## 파일 구조

```
├── README.md
├── requirements.txt
├── preprocess.py       # 데이터 전처리
├── train.py            # 모델 학습
├── predict.py          # 추론
```

---

## 실행 방법

**설치**
```bash
pip install -r requirements.txt
```

**학습**
```bash
python train.py
```

**추론 (대화형)**
```bash
python predict.py
```

**추론 (배치)**
```bash
python predict.py --file test.csv
```

---

## 학습 설정

| 항목 | 값 |
|------|----|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Max Epochs | 10 |
| Early Stopping | patience 3 |
| Max Length | 128 |
| Class Imbalance | Weighted Cross Entropy |

---

## 추론 방식

입력 텍스트를 `kss`로 문장 단위로 분리한 후 문장별 감정 확률을 평균내어 최종 감정을 분류합니다.

```
입력 텍스트
    ↓ kss 문장 분리
[문장1, 문장2, 문장3, ...]
    ↓ 각 문장 개별 추론
[확률 벡터1, 확률 벡터2, 확률 벡터3, ...]
    ↓ 평균
최종 감정 분류 결과
```

---

## 향후 개선 계획

- max_length 512로 재학습
- 하이퍼파라미터 튜닝
- 소분류 58개 클래스 확장 실험

---

## 데이터셋 출처

AI Hub 감성대화 말뭉치
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86
