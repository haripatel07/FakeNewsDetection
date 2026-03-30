# Fake News Detection

This repository contains a notebook-based model training pipeline for fake news detection using a text classification model.  
The model takes raw article text (and optional title) and classifies it as `REAL` or `FAKE`.

## Setup

1. Create virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train.py --data news.csv --model ./model/fake_news_detector.pkl
```

3. Run the API locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Deployment

Build and run the Docker image:

```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

---

## API Reference

- `POST /predict`
  - request: `{"text": "article body...", "title": "optional title"}`
  - response: `{"label": "FAKE", "confidence": 0.92}`

- `GET /health`
  - response: `{"status": "ok"}`

---

## Model Info

- Model type: `Pipeline` with `TfidfVectorizer` + `PassiveAggressiveClassifier` (from notebook training flow)
- Dataset: `news.csv` (contains `text` and `label` columns)
- Preprocessing:
  - Remove non-alphanumeric characters
  - Lowercase
  - Lemmatization using `WordNetLemmatizer`

- Performance: model uses `accuracy_score` on 80/20 split. In notebook example, `PassiveAggressiveClassifier` reached ~91+% (exact value depends on data and random seed).

---

## Notes

- Existing notebook (`FakeNewsDetection.ipynb`) holds exploratory analysis and model selection.
- for production, we now use `app.py` for inference and `Dockerfile` for containerization.
