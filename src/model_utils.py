import os
import re
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def _safe_lemmatize_tokens(tokens):
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except Exception:
        return tokens


def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().strip()
    tokens = text.split()
    tokens = _safe_lemmatize_tokens(tokens)
    return " ".join(tokens)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
        ("clf", PassiveAggressiveClassifier(max_iter=50, random_state=42))
    ])


def train_and_save_model(csv_path: str, model_path: str = "./model/fake_news_detector.pkl") -> Tuple[float, Pipeline]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    return acc, pipeline


def load_model(model_path: str = "./model/fake_news_detector.pkl") -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
    return joblib.load(model_path)


def predict_text(model, text: str, title: str = "", threshold: float = 0.5) -> Tuple[str, float]:
    combined = f"{title or ''} {text or ''}".strip()
    preprocessed = preprocess_text(combined)

    label = model.predict([preprocessed])[0]

    confidence = 0.0
    if hasattr(model, "decision_function"):
        raw = model.decision_function([preprocessed])[0]
        # map decision margin to probability-like score
        confidence = 1 / (1 + np.exp(-raw)) if label == "REAL" else 1 - (1 / (1 + np.exp(-raw)))
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba([preprocessed])[0]
        classes = model.classes_
        if "REAL" in classes and "FAKE" in classes:
            idx = list(classes).index(label)
            confidence = float(probs[idx])
        else:
            confidence = float(max(probs))
    else:
        confidence = 1.0

    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence < threshold:
        # fallback label to existing prediction still but with low confidence
        pass

    return label, confidence
