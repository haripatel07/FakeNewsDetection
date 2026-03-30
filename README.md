# Fake News Detection

This repository contains a complete ML project for fake news classification, ending in a deployable FastAPI inference service.

- Data source: `news.csv` (text + label)
- Best model: TF-IDF + PassiveAggressiveClassifier
- Output: `REAL` / `FAKE` with confidence score

## Project structure

- `app.py` : FastAPI app entrypoint (`/predict`, `/health`)
- `src/model_utils.py` : preprocessing, training, loading, prediction helpers
- `src/train.py` : train pipeline and model serialization
- `model/fake_news_detector.pkl` : trained model artifact
- `requirements.txt` : dependencies
- `Dockerfile` : container build
- `.env.example` : runtime env var template
- `FakeNewsDetection.ipynb` : exploratory EDA and model comparison

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --break-system-packages -r requirements.txt
```

## Training

```bash
python src/train.py --data news.csv --model ./model/fake_news_detector.pkl
```

The training pipeline includes:

1. read `news.csv`
2. clean text `[^a-zA-Z0-9\s]` + lowercasing
3. optional lemmatization fallback when WordNet is unavailable
4. `TfidfVectorizer(stop_words='english', max_df=0.7)`
5. `PassiveAggressiveClassifier(max_iter=50)`
6. save model with `joblib`

## Run local API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## .env example

```ini
PORT=8000
MODEL_PATH=./model/fake_news_detector.pkl
CONFIDENCE_THRESHOLD=0.5
```

## API Reference

### POST /predict

Request body:

```json
{
  "text": "article body...",
  "title": "optional title"
}
```

Response:

```json
{
  "label": "REAL" | "FAKE",
  "confidence": 0.92
}
```

### GET /health

Response:

```json
{"status": "ok"}
```

## Docker

Build and run:

```bash
docker build -t fake-news-detector .

docker run -p 8000:8000 fake-news-detector
```

## Model Info

- Model pipeline: TF-IDF vectorizer + PassiveAggressiveClassifier
- Input features: cleaned & lemmatized text
- Output: binary label with confidence probability (from decision score)
- Reported notebook accuracy: 93%+ (for sample split)

## Validation

- `python src/train.py` builds/overwrites `model/fake_news_detector.pkl`
- `app.py` loads model on startup with `load_model`
- `predict_text` returns label/confidence
- `POST /predict` and `GET /health` are available

## Next recommended improvements

- add tests (`pytest`, `tests/`)
- add CI pipeline (GitHub Actions/Bamboo/etc.)
- add robustness to malformed or large inputs
- add logging and metrics (`/metrics` Prometheus)
