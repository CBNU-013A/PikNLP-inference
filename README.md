# PikNLP-server

[PikNLP](https://github.com/CBNU-013A/PikNLP)(ELECTRA 기반 자연어 처리 모델)을 서빙하는 FastAPI 서버

## 실행방법

> [!warning]
> Docker를 사용하지 않는 경우, [uv](https://github.com/astral-sh/uv)가 필요합니다.

### uv 환경 구성
```bash
uv sync
uv run run.py
```

### Docker

GPU(pytorch:2.7.0-cuda12.6-cudnn9-runtime) 사용

```bash
docker build -t piknlp-server .
docker run --gpus all -p <port> --env-file .env piknlp-server:latest
```


## 📡 API 명세

### 1. `GET /health`

- 서버 및 모델 상태 확인

#### ✅ 응답 예시
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "cuda_available": true,
  "API-MODE": "production"
}
```

### 2. `POST /api/v1/predict`
- 리뷰 텍스트에 대한 감성 예측 수행

#### 📥 요청 형식 (JSON)
```json
{
  "text": "날씨도 좋고 조용해서 힐링됐어요"
}
```

#### 📤 응답 형식

> [!Note] 
> 분석 카테고리는 지속적으로 개선 예정입니다.

```json
{
  "sentiments": {
    "주차": "neg",
    "교통편": "pos",
    "청결/관리": "pos",
    "혼잡도": "none",
    "편의시설": "none",
    "가격": "none",
    "동반": "none",
    "장소": "pos",
    "활동": "none"
  },
  "categories": {
    "동반": "혼자",
    "시점": "none",
    "장소": "자연경관",
    "활동": "탐방"
  }
}

```

### 3. `GET /api/v1/categories`

- 모델에서 분석할 카테고리 목록 확인

#### ✅ 응답 예시

```json
{
  "sentiment_model": [
    "주차",
    "교통편",
    "청결/관리",
    "혼잡도",
    "편의시설",
    "가격",
    "동반",
    "장소",
    "활동"
  ],
  "category_map": {
    "동반": [
      "혼자",
      "가족",
      "연인",
      "친구",
      "반려동물",
      "단체",
      "none"
    ],
    "시점": [
      "봄",
      "여름",
      "가을",
      "겨울",
      "주간",
      "야간",
      "none"
    ],
    "장소": [
      "자연경관",
      "도시명소",
      "문화역사",
      "상업",
      "휴양",
      "none"
    ],
    "활동": [
      "탐방",
      "관람",
      "참여",
      "먹거리",
      "쇼핑",
      "포토존",
      "none"
    ]
  }
}
```