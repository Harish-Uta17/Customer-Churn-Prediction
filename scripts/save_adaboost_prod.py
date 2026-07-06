#!/usr/bin/env python
"""Train AdaBoost on the project's pipeline and save as the production artifact.

Overwrites: models/churn_model_best.pkl and models/churn_model_best_metadata.json
"""
from pathlib import Path
from src.config import ConfigManager
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner, clean_data
from src.preprocessing.preprocessor import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.model_manager import save_trained_model
from sklearn.model_selection import train_test_split


def main():
    config = ConfigManager()
    data_path = Path(config.get("data.raw_path", "data/raw/churn.csv"))
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parents[1] / data_path

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    df = DataLoader.load_csv(str(data_path))
    df = clean_data(df)
    df, _ = DataCleaner.encode_target(df, 'Churn')

    X, y = FeatureEngineer.prepare_features(df, 'Churn')
    X = FeatureEngineer.encode_categorical_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, scaler = FeatureEngineer.scale_features(X_train, X_test, return_scaler=True)
    X_train, y_train = FeatureEngineer.handle_class_imbalance(X_train, y_train)

    # Train only AdaBoost
    model = ModelTrainer.train_multiple_models(X_train, y_train)['AdaBoost']

    # Evaluate on test set
    from src.models.evaluate import ModelEvaluator
    metrics = ModelEvaluator.evaluate_model(model, X_test, y_test, model_name='AdaBoost')

    # Save as churn_model_best
    feature_names = X.columns.tolist()
    save_trained_model(model, scaler, feature_names, metrics, model_name='churn_model_best')
    print('Saved AdaBoost as models/churn_model_best.pkl')


if __name__ == '__main__':
    main()
