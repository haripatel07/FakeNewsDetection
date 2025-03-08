# Fake News Detection

## Overview
This project develops a machine learning system to detect fake news by analyzing text content. It compares various text vectorization techniques and models to identify the most effective approach for distinguishing real from fake news, aiming to enhance content verification.

## Aim
- Evaluate combinations of vectorizers (CountVectorizer, TfidfVectorizer, HashingVectorizer, Doc2Vec) and models (Naive Bayes, PassiveAggressiveClassifier, SGDClassifier, Logistic Regression).
- Refine the best model using error analysis and cross-validation.

## Dataset
- **news.csv**: 6,335 articles labeled "REAL" (3,171) or "FAKE" (3,164).

## Installation
```bash
pip install numpy pandas scikit-learn gensim nltk matplotlib seaborn textblob scipy
```
Download NLTK data:
```python
import nltk
nltk.download('wordnet')
```

## Usage
Clone the repository:
```bash
git clone https://github.com/haripatel07/FakeNewsDetection
cd FakeNewsDetection
```
Add `news.csv` to the directory.
Run the script:
```bash
python fake_news_detection.py
```

## Methodology
### Preprocessing:
- Removed punctuation
- Converted text to lowercase
- Applied lemmatization

### Models & Performance:
| Vectorizer + Model                          | Accuracy |
|---------------------------------------------|----------|
| CountVectorizer + Naive Bayes               | 89.58%   |
| **TfidfVectorizer + PassiveAggressiveClassifier** | **93.21% (Best)** |
| HashingVectorizer + SGDClassifier           | 91.48%   |
| Doc2Vec + Logistic Regression               | 86.82%   |

### Improvements:
- Added **sentiment analysis** (TextBlob)
- Applied **hyperparameter tuning** (GridSearchCV)
- Cross-Validation Accuracy: **93.33% (±0.86%)** post-improvement

## Results
### Best Model: **TfidfVectorizer + PassiveAggressiveClassifier (93.21%)**

#### Confusion Matrix:
```
[[586, 42],
 [44, 595]]
```

#### Classification Report:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| FAKE  | 0.93      | 0.93   | 0.93     |
| REAL  | 0.93      | 0.93   | 0.93     |

## Next Steps
- Analyze misclassifications further.
- Explore additional features (e.g., Named Entity Recognition - NER).
- Test ensemble or deep learning models.

## License
[MIT License](LICENSE.md)

