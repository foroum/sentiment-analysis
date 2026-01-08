import argparse
import json
import os

from datasets import load_dataset
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB # naive bayes
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    MAX_FEATURES,
    NGRAM_RANGE,
    DEFAULT_MODEL_PATH,
    METRICS_PATH,
)


def build_models(max_features: int, ngram_range):
    """
    Build a dict of model_name -> sklearn Pipeline(TfidfVectorizer + classifier).
    All classifiers support predict_proba thanks to CalibratedClassifierCV for LinearSVC.
    """
    models = {}

    # Shared TF-IDF config
    tfidf_params = {
        "max_features": max_features,
        "ngram_range": ngram_range,
    }

    # 1) Logistic Regression
    models["logreg"] = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)),
        ]
    )

    # 2) Naive Bayes
    models["naive_bayes"] = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB()),
        ]
    )

    # 3) Linear SVM with probability calibration
    svm = LinearSVC(random_state=RANDOM_STATE)
    calibrated_svm = CalibratedClassifierCV(svm, cv=3)

    models["linear_svm"] = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", calibrated_svm),
        ]
    )

    return models


def train_and_evaluate_models(
    model_dict,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    """
    Fit each model, compute val/test accuracy, and keep metrics.
    Returns:
      results: dict[model_name] = {"val_accuracy", "test_accuracy", "report", "pipeline"}
      best_name: name of model with highest val_accuracy
    """
    results = {}

    for name, pipeline in model_dict.items():
        print(f"\nTraining model... {name}")
        pipeline.fit(X_train, y_train)

        # Validation
        val_preds = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        # Test
        test_preds = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        report = classification_report(
            y_test, test_preds, target_names=["NEGATIVE", "POSITIVE"], digits=4
        )

        print(f"Validation accuracy ({name}): {val_acc:.4f}")
        print(f"Test accuracy ({name}): {test_acc:.4f}")

        results[name] = {
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "report": report,
            "pipeline": pipeline,
        }

    # Pick the model with best validation accuracy
    best_name = max(results.keys(), key=lambda m: results[m]["val_accuracy"])
    return results, best_name


def save_metrics(results, best_name, metrics_path: str):
    """
    Save metrics (without the pipeline objects) as JSON.
    """
    # Remove the actual sklearn pipelines (not JSON serializable)
    json_friendly = {}
    for name, info in results.items():
        json_friendly[name] = {
            "val_accuracy": info["val_accuracy"],
            "test_accuracy": info["test_accuracy"],
            "report": info["report"],
        }
    payload = {
        "best_model": best_name,
        "models": json_friendly,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n Saved metrics to {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multiple sentiment models on the IMDB dataset and save the best one."
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help=f"Max number of TF-IDF features (default: {MAX_FEATURES})",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=NGRAM_RANGE[0],
        help=f"Minimum n-gram size (default: {NGRAM_RANGE[0]})",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=NGRAM_RANGE[1],
        help=f"Maximum n-gram size (default: {NGRAM_RANGE[1]})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Validation split size from train set (default: {TEST_SIZE})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save the best model pipeline (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=METRICS_PATH,
        help=f"Path to save metrics JSON (default: {METRICS_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ngram_range = (args.ngram_min, args.ngram_max)
    print("Configuration:")
    print(f"  - max_features: {args.max_features}")
    print(f"  - ngram_range: {ngram_range}")
    print(f"  - val_size (from train): {args.test_size}")
    print(f"  - model_path: {args.model_path}")
    print(f"  - metrics_path: {args.metrics_path}")

    # 1) Load IMDB
    print("\n Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # 2) Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        train_labels,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=train_labels,
    )

    print(
        f"Sizes → train: {len(X_train)}, val: {len(X_val)}, test: {len(test_texts)}"
    )

    # 3) Build models
    model_dict = build_models(args.max_features, ngram_range)

    # 4) Train & evaluate
    results, best_name = train_and_evaluate_models(
        model_dict,
        X_train,
        y_train,
        X_val,
        y_val,
        test_texts,
        test_labels,
    )

    # 5) Save best model pipeline
    best_pipeline = results[best_name]["pipeline"]
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    dump(best_pipeline, args.model_path)
    print(f"\nBest model: {best_name}")
    print(f"Saved best model pipeline to: {args.model_path}")

    # 6) Save metrics JSON
    save_metrics(results, best_name, args.metrics_path)


if __name__ == "__main__":
    main()
