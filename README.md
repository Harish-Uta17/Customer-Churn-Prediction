# Customer Churn Prediction

Professional machine learning project for predicting telecom customer churn and presenting the results through a polished Streamlit dashboard.

## Overview

This project trains a churn classification model on the Telco Customer Churn dataset, evaluates multiple algorithms, saves the best-performing artifact, and serves predictions in a modern web UI.

The current production model is AdaBoost, selected by ROC-AUC.

Current evaluation snapshot (top model — AdaBoost):

- ROC-AUC: 0.8637
- Accuracy: 0.7821
- Precision: 0.5632
- Recall: 0.7882

## What The Project Does

- Loads and cleans the churn dataset.
- Encodes categorical features and prepares a training matrix.
- Handles class imbalance with SMOTE.
- Scales features before training.
- Compares multiple classification models.
- Saves the best model, scaler, feature metadata, and metrics.
- Provides a Streamlit app for dashboard exploration and single-customer churn prediction.

## Repository Structure

```text
Customer_Churn_prediction/
├── app/
│   └── streamlit_app.py         # Streamlit inference app and dashboard
├── config/
│   ├── config.yaml              # Main configuration used by the pipeline
│   ├── dev.yaml                 # Development configuration
│   └── production.yaml          # Production configuration
├── data/
│   ├── raw/
│   │   └── churn.csv            # Training dataset
│   └── processed/
│       └── churn_processed.csv  # Cleaned dataset written by the pipeline
├── images/
│   ├── confusion_matrix.png     # Documentation / report image
│   ├── feature_importance.png   # Documentation / report image
│   └── roc_curve.png            # Documentation / report image
├── logs/                        # Runtime logs
├── models/                      # Saved model artifacts
├── notebook/
│   └── Customer_Churn_Prediction.ipynb
├── src/
│   ├── config.py                # Configuration loader
│   ├── data/
│   │   ├── cleaner.py           # Data cleaning utilities
│   │   └── loader.py            # Dataset loading helpers
│   ├── preprocessing/
│   │   └── preprocessor.py      # Encoding, feature engineering, scaling helpers
│   ├── models/
│   │   ├── evaluate.py          # Model comparison and evaluation
│   │   ├── model_manager.py     # Save/load model artifacts
│   │   ├── predict.py           # Prediction utilities
│   │   └── train.py             # Model training routines
│   └── utils/
│       ├── constants.py         # Shared constants
│       ├── helpers.py           # Helper functions
│       └── logger.py            # Logging setup
├── main.py                      # End-to-end training pipeline
├── quick_test.py                # Validation script for the repository
├── README.md                    # Project documentation
├── requirements.txt             # Runtime dependencies
├── setup.py                     # Package metadata and install configuration
└── Dockerfile                   # Container build definition
```

## Project Flow

1. Load configuration from `config/config.yaml`.
2. Read the raw dataset from `data/raw/churn.csv`.
3. Clean data, fix types, and remove inconsistencies.
4. Encode the target and one-hot encode categorical features.
5. Split data into train and test sets.
6. Balance the training split with SMOTE.
7. Scale features using `StandardScaler`.
8. Train multiple models.
9. Compare metrics and select the best model.
10. Save the model artifact and metadata into `models/`.

## Dataset

The repository includes the dataset required to train and test the project:

- `data/raw/churn.csv`

The training and dashboard flow expect this file to be present.

## Streamlit Application

Run the application with:

```bash
streamlit run app/streamlit_app.py
```

The app includes:

- A Home page with project overview and model summary.
- A Dashboard page that loads the dataset and shows churn statistics.
- A Predict Churn page for manual customer scoring.
- An Information page with project background and feature context.

## Training Pipeline

Run the full training workflow with:

```bash
python main.py
```

This executes the full pipeline and saves the best model to `models/churn_model_best.pkl`.

Pipeline outputs include:

- The fitted model.
- The fitted scaler.
- Feature names used during training.
- Evaluation metrics.
- JSON metadata for readability.

## Validation Script

Run the repository validation script with:

```bash
python quick_test.py
```

This checks key pieces of the project such as environment imports, configuration loading, data loading, cleaning, preprocessing, and model workflow assumptions.

## Configuration

Primary configuration lives in `config/config.yaml`.

Important settings include:

- `data.raw_path` - source dataset location.
- `data.test_size` - test split ratio.
- `preprocessing.handle_imbalance` - enables SMOTE.
- `preprocessing.scaling_method` - scaling strategy.
- `model.type` - selected model family.
- `evaluation.primary_metric` - model ranking metric.
- `logging.log_file` - runtime log file path.

## Saved Artifacts

The training pipeline writes artifacts to `models/`.

Typical outputs:

- `models/churn_model_best.pkl`
- `models/churn_model_best_metadata.json`

These artifacts are used directly by the Streamlit app.

## Documentation Assets

The `images/` directory contains the visuals you built for the project and can be embedded in the README or reused in reports:

- `confusion_matrix.png`
- `feature_importance.png`
- `roc_curve.png`

### Visuals

## ROC Curve

![ROC Curve](https://raw.githubusercontent.com/Harish-Uta17/Customer-Churn-Prediction/main/images/roc_curve.png)

## Feature Importance

![Feature Importance](https://raw.githubusercontent.com/Harish-Uta17/Customer-Churn-Prediction/main/images/feature_importance.png)

## Confusion Matrix

![Confusion Matrix](https://raw.githubusercontent.com/Harish-Uta17/Customer-Churn-Prediction/main/images/confusion_matrix.png)

Keep these if you want the repository to remain presentation-ready.

## Operations Performed

The project pipeline performs the following operations end to end:

1. Load the customer churn dataset from `data/churn.csv`.
2. Clean and standardize column values.
3. Fix data types and handle missing values.
4. Encode the target column and prepare categorical features.
5. Split the data into training and testing sets.
6. Balance the training split with SMOTE.
7. Scale the features before model training.
8. Train multiple classification models.
9. Compare model metrics and select the best model.
10. Save the trained model, scaler, feature metadata, and evaluation metrics.

## Notebook

The notebook under `notebook/Customer_Churn_Prediction.ipynb` is exploratory.

Recommendation:

- Keep it if you want to preserve analysis history and experimentation notes.
- Remove it from a production-only branch if you want a lean deployment repository.

## Dependencies

Runtime dependencies are listed in `requirements.txt`.

Main packages used by the project:

- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- PyYAML
- scipy
- plotly
- streamlit
- matplotlib
- seaborn

## Installation

Create a virtual environment and install the requirements:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python main.py
```

### 2. Run the app

```bash
streamlit run app/streamlit_app.py
```

### 3. Run validation

```bash
python quick_test.py
```

## Deployment Notes

The project is compatible with Streamlit Cloud, Render, Railway, or similar Python hosting platforms.

Before deployment, make sure:

- The trained model artifact exists in `models/`.
- `data/raw/churn.csv` is available if you want the dashboard to auto-load the dataset.
- The required Python dependencies are installed.

## License

Add your license here if you intend to publish the repository publicly.