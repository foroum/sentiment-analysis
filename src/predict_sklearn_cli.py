import argparse
import os
import sys

from joblib import load

from src.config import DEFAULT_MODEL_PATH, NEUTRAL_THRESHOLD

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run 'python -m src.train_sklearn_imdb' first."
        )
    return load(model_path)


def predict_text(text: str, model, neutral_threshold: float):
    # model is a Pipeline [TfidfVectorizer -> classifier]
    proba = model.predict_proba([text])[0]
    pred_idx = proba.argmax()
    confidence = float(proba[pred_idx])

    if confidence < neutral_threshold:
        label = "NEUTRAL / UNCERTAIN"
    else:
        label = "POSITIVE" if pred_idx == 1 else "NEGATIVE"

    return label, confidence, proba


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict sentiment for input text using the trained IMDB model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model pipeline (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=NEUTRAL_THRESHOLD,
        help=f"Confidence threshold for NEUTRAL / UNCERTAIN (default: {NEUTRAL_THRESHOLD})",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to classify. If omitted, will prompt for input.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.text:
        text = " ".join(args.text)
    else:
        text = input("Enter a movie review: ")

    print(f"\nLoading model from {args.model_path} ...")
    model = load_model(args.model_path)

    label, confidence, proba = predict_text(text, model, args.neutral_threshold)

    print("\n - Review:")
    print(text)
    print("\n - Prediction:")
    print(f"  Label: {label}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Probabilities [NEGATIVE, POSITIVE]: {proba}")


if __name__ == "__main__":
    main()
