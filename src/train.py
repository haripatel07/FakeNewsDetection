import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_utils import train_and_save_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Fake News Detector model")
    parser.add_argument("--data", type=str, default="news.csv", help="Path to the CSV dataset")
    parser.add_argument("--model", type=str, default="./model/fake_news_detector.pkl", help="Path to save the trained model")
    args = parser.parse_args()

    accuracy, _ = train_and_save_model(args.data, args.model)
    print(f"Model trained and saved to {args.model} with accuracy {accuracy*100:.2f}%")
