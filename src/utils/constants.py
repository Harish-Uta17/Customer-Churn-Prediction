"""
Constants and Configuration

All magic numbers and constants defined here for easy modification.
This prevents "magic numbers" scattered throughout the codebase.
"""

# File Paths
DATA_DIR = "data"
RAW_DATA_FILE = "churn.csv"
PROCESSED_DATA_FILE = "churn_processed.csv"

MODELS_DIR = "models"
LOGS_DIR = "logs"

# Data Splitting
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing
SCALING_METHOD = "standard"  # "standard" or "minmax"
HANDLE_MISSING = "median"    # "median", "mean", or "drop"
HANDLE_OUTLIERS = True
OUTLIER_THRESHOLD = 3.0  # Standard deviations

# SMOTE Configuration
USE_SMOTE = True
SMOTE_RANDOM_STATE = 42

# Feature Engineering
DROP_COLUMNS = ['customerID']
TARGET_COLUMN = 'Churn'
CATEGORICAL_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Model Configuration
MODEL_TYPE = "logistic_regression"  # Model to use
MODELS_TO_COMPARE = [
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'adaboost'
]

# Hyperparameters - Logistic Regression (Best Model)
LR_HYPERPARAMS = {
    'C': 10,
    'solver': 'liblinear',
    'max_iter': 500,
    'random_state': RANDOM_STATE
}

# Hyperparameters - Gradient Boosting
GB_HYPERPARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_split': 5,
    'subsample': 0.8,
    'random_state': RANDOM_STATE
}

# Evaluation Metrics
EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
PRIMARY_METRIC = 'roc_auc'  # Main metric to optimize

# Logging Configuration
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE = "logs/app.log"

# Streamlit Configuration
STREAMLIT_MAX_FILE_SIZE = 200  # MB
STREAMLIT_LAYOUT = "wide"
STREAMLIT_THEME = "light"

# Prediction Confidence Threshold
CHURN_RISK_THRESHOLD = 0.5  # Probability threshold for churn prediction
CONFIDENCE_THRESHOLD_HIGH = 0.7
CONFIDENCE_THRESHOLD_LOW = 0.3
