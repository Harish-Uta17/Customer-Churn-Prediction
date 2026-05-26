#!/usr/bin/env python
"""Compare multiple models on the churn dataset and report metrics.

Usage: python scripts/compare_models.py
"""
from pathlib import Path

import sys
from pathlib import Path
import traceback

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.data.loader import DataLoader
    from src.data.cleaner import DataCleaner, clean_data
    from src.preprocessing.preprocessor import FeatureEngineer
    from src.models.train import ModelTrainer
    from src.models.evaluate import ModelEvaluator
    from sklearn.model_selection import train_test_split
except Exception:
    print("Failed to import project modules:")
    traceback.print_exc()
    raise


def main():
    data_path = Path("data/churn.csv")
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path} — please ensure data/churn.csv exists")

    df = DataLoader.load_csv(str(data_path))
    df = clean_data(df)
    df, _ = DataCleaner.encode_target(df, 'Churn')

    X, y = FeatureEngineer.prepare_features(df, 'Churn')
    X = FeatureEngineer.encode_categorical_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, scaler = FeatureEngineer.scale_features(X_train, X_test, return_scaler=True)
    X_train, y_train = FeatureEngineer.handle_class_imbalance(X_train, y_train)

    print("Training multiple models (this may take a short while)...")
    models = ModelTrainer.train_multiple_models(X_train, y_train)

    print("Evaluating models on the test set...")
    comparison_df = ModelEvaluator.compare_models(models, X_test, y_test)

    out_path = Path("models")
    out_path.mkdir(exist_ok=True)
    csv_path = out_path / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    print(f"Comparison written to: {csv_path}")
    best = comparison_df.iloc[0]
    print(f"\nTop model: {best['Model']} — ROC-AUC: {best.get('ROC-AUC', 'N/A')}")


if __name__ == '__main__':
    main()
