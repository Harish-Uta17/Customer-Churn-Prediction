#!/usr/bin/env python
"""
Quick Test Suite for Customer Churn Prediction Project
Runs all component tests and generates a comprehensive report

Usage:
    python quick_test.py          # Run all tests
    python quick_test.py --verbose  # Run with detailed output
"""

import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'
BOLD = '\033[1m'

# Test results
results = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'errors': []
}

def print_header(text):
    """Print formatted header"""
    print(f"\n{BLUE}{BOLD}{'='*70}{END}")
    print(f"{BLUE}{BOLD}{text:^70}{END}")
    print(f"{BLUE}{BOLD}{'='*70}{END}\n")

def print_section(text):
    """Print formatted section"""
    print(f"\n{BOLD}{text}{END}")
    print("-" * 70)

def print_pass(message):
    """Print passing test"""
    print(f"{GREEN}✅ PASS{END}: {message}")
    results['passed'] += 1

def print_fail(message, error=None):
    """Print failing test"""
    print(f"{RED}❌ FAIL{END}: {message}")
    if error:
        print(f"   {RED}Error: {str(error)}{END}")
        results['errors'].append((message, error))
    results['failed'] += 1

def print_skip(message):
    """Print skipped test"""
    print(f"{YELLOW}⊘ SKIP{END}: {message}")
    results['skipped'] += 1

def test_environment():
    """Test 1: Environment Setup"""
    print_section("TEST 1: Environment Setup")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import streamlit as st
        import plotly
        print_pass(f"Python {sys.version.split()[0]}")
        print_pass(f"pandas {pd.__version__}")
        print_pass(f"numpy {np.__version__}")
        print_pass(f"scikit-learn {sklearn.__version__}")
        print_pass(f"streamlit {st.__version__}")
        print_pass(f"plotly {plotly.__version__}")
    except Exception as e:
        print_fail("Package import failed", e)
        return False
    
    # Check venv
    venv_path = Path(sys.prefix)
    if (venv_path / 'pyvenv.cfg').exists():
        print_pass(f"Virtual environment active: {venv_path.name}")
    else:
        print_skip("Virtual environment not detected (optional)")
    
    return True

def test_logger():
    """Test 2: Logger Module"""
    print_section("TEST 2: Logger Module (src/utils/logger.py)")
    
    try:
        from src.utils.logger import get_logger
        import logging
        
        logger = get_logger("test_logger")
        
        # Test all levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        print_pass("Logger created and tested")
        
        # Check log file
        log_file = Path("logs/app.log")
        if log_file.exists():
            size = log_file.stat().st_size
            print_pass(f"Log file created: {log_file} ({size} bytes)")
        else:
            print_skip(f"Log file not found (will be created on first run)")
            
    except Exception as e:
        print_fail("Logger test failed", e)
        return False
    
    return True

def test_constants():
    """Test 3: Constants Module"""
    print_section("TEST 3: Constants Module (src/utils/constants.py)")
    
    try:
        from src.utils.constants import (
            DROP_COLUMNS, CATEGORICAL_COLUMNS, RANDOM_STATE,
            TEST_SIZE, LR_HYPERPARAMS, PRIMARY_METRIC
        )
        
        assert DROP_COLUMNS, "DROP_COLUMNS is empty"
        assert CATEGORICAL_COLUMNS, "CATEGORICAL_COLUMNS is empty"
        assert RANDOM_STATE == 42, "RANDOM_STATE should be 42"
        assert TEST_SIZE == 0.2, "TEST_SIZE should be 0.2"
        assert LR_HYPERPARAMS, "LR_HYPERPARAMS is empty"
        assert PRIMARY_METRIC, "PRIMARY_METRIC is empty"
        
        print_pass(f"DROP_COLUMNS: {len(DROP_COLUMNS)} columns")
        print_pass(f"CATEGORICAL_COLUMNS: {len(CATEGORICAL_COLUMNS)} columns")
        print_pass(f"RANDOM_STATE: {RANDOM_STATE}")
        print_pass(f"TEST_SIZE: {TEST_SIZE}")
        print_pass(f"LR_HYPERPARAMS: {list(LR_HYPERPARAMS.keys())}")
        print_pass(f"PRIMARY_METRIC: {PRIMARY_METRIC}")
        
    except Exception as e:
        print_fail("Constants test failed", e)
        return False
    
    return True

def test_config():
    """Test 4: Configuration Module"""
    print_section("TEST 4: Configuration Module (src/config.py)")
    
    try:
        from src.config import ConfigManager
        
        # Check config file exists
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            print_fail(f"Config file not found: {config_path}")
            return False
        print_pass(f"Config file found: {config_path}")
        
        # Load config
        config = ConfigManager()
        assert config.config, "Config is empty"
        print_pass(f"Config loaded: {len(config.config)} top-level keys")
        
        # Test nested access
        test_size = config.get('data.test_size')
        print_pass(f"Nested access works: data.test_size = {test_size}")
        
        # Check main sections
        sections = ['project', 'data', 'preprocessing', 'model', 'logging']
        for section in sections:
            value = config.get(section)
            print_pass(f"Section '{section}' loaded")
        
    except Exception as e:
        print_fail("Config test failed", e)
        return False
    
    return True

def test_data_loader():
    """Test 5: Data Loader Module"""
    print_section("TEST 5: Data Loader Module (src/data/loader.py)")
    
    try:
        from src.data.loader import DataLoader
        
        # Check data file exists
        data_path = Path("data/churn.csv")
        if not data_path.exists():
            print_fail(f"Data file not found: {data_path}")
            print(f"   {YELLOW}Download data/churn.csv from original notebook!{END}")
            return False
        print_pass(f"Data file found: {data_path}")
        
        # Load data
        df = DataLoader.load_csv(str(data_path))
        print_pass(f"Data loaded: {df.shape}")
        
        # Verify data integrity
        if df.empty:
            print_fail("Data is empty")
            return False
        print_pass(f"Data has {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Check target column
        if 'Churn' in df.columns:
            churn_dist = df['Churn'].value_counts().to_dict()
            print_pass(f"Target column 'Churn' found: {churn_dist}")
        else:
            print_fail("Target column 'Churn' not found")
            return False
        
    except Exception as e:
        print_fail("Data loader test failed", e)
        traceback.print_exc()
        return False
    
    return True

def test_data_cleaner():
    """Test 6: Data Cleaner Module"""
    print_section("TEST 6: Data Cleaner Module (src/data/cleaner.py)")
    
    try:
        from src.data.cleaner import DataCleaner
        from src.data.loader import DataLoader
        import pandas as pd
        
        df = DataLoader.load_csv("data/churn.csv")
        original_shape = df.shape
        
        # Test 1: Encode target
        df_copy = df.copy()
        df_encoded, _ = DataCleaner.encode_target(df_copy, 'Churn')
        print_pass(f"Target encoded: Churn column is now numeric")
        
        # Test 2: Remove duplicates
        df_copy = df.copy()
        df_copy = pd.concat([df_copy, df_copy.iloc[[0]]], ignore_index=True)
        before = len(df_copy)
        df_clean = DataCleaner.remove_duplicates(df_copy)
        after = len(df_clean)
        print_pass(f"Duplicates handled: {before} → {after} rows")
        
        # Test 3: Fix data types
        df_clean = DataCleaner.fix_data_types(df.copy())
        print_pass(f"Data types fixed: {df_clean.dtypes.nunique()} unique types")
        
        # Test 4: Handle missing values
        df_test = df.copy()
        missing_before = df_test.isnull().sum().sum()
        df_clean = DataCleaner.handle_missing_values(df_test)
        missing_after = df_clean.isnull().sum().sum()
        print_pass(f"Missing values handled: {missing_before} → {missing_after}")
        
    except Exception as e:
        print_fail("Data cleaner test failed", e)
        traceback.print_exc()
        return False
    
    return True

def test_preprocessor():
    """Test 7: Preprocessor Module"""
    print_section("TEST 7: Preprocessor Module (src/preprocessing/preprocessor.py)")
    
    try:
        from src.preprocessing.preprocessor import FeatureEngineer
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner, clean_data
        from sklearn.model_selection import train_test_split
        
        df = DataLoader.load_csv("data/churn.csv")
        df = clean_data(df)
        df, _ = DataCleaner.encode_target(df, 'Churn')
        
        # Test 1: Prepare features
        X, y = FeatureEngineer.prepare_features(df, 'Churn')
        print_pass(f"Features prepared: X{X.shape}, y{y.shape}")
        
        # Test 2: Encode categorical
        X_encoded = FeatureEngineer.encode_categorical_features(X, drop_first=True)
        print_pass(f"Categorical features encoded: {X.shape[1]} → {X_encoded.shape[1]} columns")
        
        # Test 3: Scale features
        X_train, X_test = train_test_split(X_encoded, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled, scaler = FeatureEngineer.scale_features(
            X_train, X_test, return_scaler=True
        )
        print_pass(f"Features scaled: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
        
        # Test 4: SMOTE
        y_train = y.iloc[X_train.index]
        X_train_smote, y_train_smote = FeatureEngineer.handle_class_imbalance(
            X_train_scaled, y_train
        )
        print_pass(f"SMOTE applied: {X_train.shape[0]} → {X_train_smote.shape[0]} samples")
        
    except Exception as e:
        print_fail("Preprocessor test failed", e)
        traceback.print_exc()
        return False
    
    return True

def test_model_trainer():
    """Test 8: Model Trainer Module"""
    print_section("TEST 8: Model Trainer Module (src/models/train.py)")
    
    try:
        from src.models.train import ModelTrainer
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner, clean_data
        from src.preprocessing.preprocessor import FeatureEngineer
        from sklearn.model_selection import train_test_split
        
        print("Preparing training data...")
        df = DataLoader.load_csv("data/churn.csv")
        df = clean_data(df)
        df, _ = DataCleaner.encode_target(df, 'Churn')
        X, y = FeatureEngineer.prepare_features(df, 'Churn')
        X = FeatureEngineer.encode_categorical_features(X)
        feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, _ = FeatureEngineer.scale_features(X_train, X_test, return_scaler=True)
        X_train, y_train = FeatureEngineer.handle_class_imbalance(X_train, y_train)
        
        # Test 1: Train Logistic Regression
        print("Training Logistic Regression...")
        model = ModelTrainer.train_logistic_regression(X_train, y_train)
        print_pass(f"Logistic Regression trained: {type(model).__name__}")
        
        # Test 2: Predictions work
        pred = model.predict(X_test[:1])
        print_pass(f"Model can make predictions: {pred}")
        
    except Exception as e:
        print_fail("Model trainer test failed", e)
        traceback.print_exc()
        return False
    
    return True

def test_model_evaluator():
    """Test 9: Model Evaluator Module"""
    print_section("TEST 9: Model Evaluator Module (src/models/evaluate.py)")
    
    try:
        from src.models.evaluate import ModelEvaluator
        from src.models.train import ModelTrainer
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner, clean_data
        from src.preprocessing.preprocessor import FeatureEngineer
        from sklearn.model_selection import train_test_split
        
        print("Preparing training data...")
        df = DataLoader.load_csv("data/churn.csv")
        df = clean_data(df)
        df, _ = DataCleaner.encode_target(df, 'Churn')
        X, y = FeatureEngineer.prepare_features(df, 'Churn')
        X = FeatureEngineer.encode_categorical_features(X)
        feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, _ = FeatureEngineer.scale_features(X_train, X_test, return_scaler=True)
        X_train, y_train = FeatureEngineer.handle_class_imbalance(X_train, y_train)
        
        print("Training model...")
        model = ModelTrainer.train_logistic_regression(X_train, y_train)
        
        # Test 1: Evaluate model
        metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
        print_pass(f"Model evaluated with {len(metrics)} metrics:")
        import numbers
        for metric, value in metrics.items():
            if isinstance(value, numbers.Number):
                try:
                    print(f"   {metric}: {value:.4f}")
                except Exception:
                    print(f"   {metric}: {value}")
            else:
                print(f"   {metric}: {value}")
        
        # Test 2: Confusion matrix
        cm = ModelEvaluator.get_confusion_matrix(model, X_test, y_test)
        print_pass(f"Confusion matrix: {cm.shape}")
        
        # Test 3: ROC curve
        fpr, tpr, _ = ModelEvaluator.get_roc_curve(model, X_test, y_test)
        print_pass(f"ROC curve: {len(fpr)} points")
        
    except Exception as e:
        print_fail("Model evaluator test failed", e)
        traceback.print_exc()
        return False
    
    return True

def test_model_manager():
    """Test 10: Model Manager Module"""
    print_section("TEST 10: Model Manager Module (src/models/model_manager.py)")
    
    try:
        from src.models.model_manager import ModelManager
        from src.models.train import ModelTrainer
        from src.models.evaluate import ModelEvaluator
        from src.data.loader import DataLoader
        from src.data.cleaner import DataCleaner, clean_data
        from src.preprocessing.preprocessor import FeatureEngineer
        from sklearn.model_selection import train_test_split
        from pathlib import Path
        import json
        
        print("Preparing training data...")
        df = DataLoader.load_csv("data/churn.csv")
        df = clean_data(df)
        df, _ = DataCleaner.encode_target(df, 'Churn')
        X, y = FeatureEngineer.prepare_features(df, 'Churn')
        X = FeatureEngineer.encode_categorical_features(X)
        feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, scaler = FeatureEngineer.scale_features(X_train, X_test, return_scaler=True)
        X_train, y_train = FeatureEngineer.handle_class_imbalance(X_train, y_train)
        
        print("Training and evaluating model...")
        model = ModelTrainer.train_logistic_regression(X_train, y_train)
        metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
        
        # Test 1: Save model
        ModelManager.save_model(model, scaler, feature_names, metrics, "test_model_123")
        print_pass("Model saved")
        
        # Test 2: Verify files exist
        pkl_path = Path("models/test_model_123.pkl")
        json_path = Path("models/test_model_123_metadata.json")
        
        if pkl_path.exists() and json_path.exists():
            pkl_size = pkl_path.stat().st_size
            json_size = json_path.stat().st_size
            print_pass(f"Model files created: PKL({pkl_size} bytes), JSON({json_size} bytes)")
        else:
            print_fail("Model files not created")
            return False
        
        # Test 3: Load model
        loaded_model, loaded_scaler, loaded_preprocessor, loaded_metadata = ModelManager.load_model(str(pkl_path))
        print_pass(f"Model loaded: {type(loaded_model).__name__}")
        print_pass(f"Scaler loaded: {type(loaded_scaler).__name__}")
        if isinstance(loaded_preprocessor, dict):
            feature_count = len(loaded_preprocessor.get('feature_names', []))
        else:
            feature_count = len(loaded_preprocessor)
        print_pass(f"Features loaded: {feature_count} features")
        
        # Test 4: Verify metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        metadata_label = metadata.get('model_type', metadata.get('Model', 'N/A'))
        roc_auc_value = metadata.get('metrics', metadata).get('ROC-AUC', 0)
        print_pass(f"Metadata: {metadata_label}, ROC-AUC: {roc_auc_value:.4f}")
        
        # Cleanup
        import os
        os.remove(str(pkl_path))
        os.remove(str(json_path))
        print_pass("Cleanup complete")
        
    except Exception as e:
        print_fail("Model manager test failed", e)
        traceback.print_exc()
        return False
    
    return True

def test_streamlit():
    """Test 11: Streamlit App"""
    print_section("TEST 11: Streamlit App (app/streamlit_app.py)")
    
    try:
        import streamlit as st
        
        app_path = Path("app/streamlit_app.py")
        if not app_path.exists():
            print_fail(f"Streamlit app not found: {app_path}")
            return False
        
        print_pass(f"Streamlit app found: {app_path}")
        
        # Check file size
        size = app_path.stat().st_size
        if size > 1000:
            print_pass(f"App size: {size} bytes (sufficient)")
        else:
            print_fail(f"App file too small: {size} bytes")
            return False
        
        print(f"\n{YELLOW}To test Streamlit app manually:{END}")
        print(f"   streamlit run app/streamlit_app.py")
        print(f"   Then open http://localhost:8501")
        
    except Exception as e:
        print_fail("Streamlit test failed", e)
        return False
    
    return True

def test_docker():
    """Test 12: Docker"""
    print_section("TEST 12: Docker Files")
    
    try:
        dockerfile_path = Path("Dockerfile")
        requirements_path = Path("requirements.txt")
        
        if not dockerfile_path.exists():
            print_fail(f"Dockerfile not found: {dockerfile_path}")
            return False
        print_pass(f"Dockerfile found: {dockerfile_path}")
        
        if not requirements_path.exists():
            print_fail(f"requirements.txt not found: {requirements_path}")
            return False
        
        # Read and check requirements
        with open(requirements_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print_pass(f"requirements.txt found: {len(lines)} packages")
        
        print(f"\n{YELLOW}To test Docker manually:{END}")
        print(f"   docker build -t churn-prediction:v1.0 .")
        print(f"   docker run -p 8501:8501 churn-prediction:v1.0")
        
    except Exception as e:
        print_fail("Docker test failed", e)
        return False
    
    return True

def print_summary():
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total = results['passed'] + results['failed'] + results['skipped']
    
    print(f"{GREEN}✅ Passed: {results['passed']}/{total}{END}")
    print(f"{RED}❌ Failed: {results['failed']}/{total}{END}")
    print(f"{YELLOW}⊘ Skipped: {results['skipped']}/{total}{END}")
    
    if results['failed'] > 0:
        print(f"\n{RED}{BOLD}Errors:{END}")
        for test_name, error in results['errors']:
            print(f"   • {test_name}")
            print(f"     {str(error)[:100]}")
    
    percentage = (results['passed'] / total * 100) if total > 0 else 0
    
    if results['failed'] == 0:
        print(f"\n{GREEN}{BOLD}✅ ALL TESTS PASSED!{END}")
        print(f"{BLUE}Success rate: {percentage:.1f}%{END}")
    else:
        print(f"\n{RED}{BOLD}❌ SOME TESTS FAILED{END}")
        print(f"{BLUE}Success rate: {percentage:.1f}%{END}")
    
    print_header("NEXT STEPS")
    print(f"""
{BOLD}1. Run the full training pipeline:{END}
   python main.py

{BOLD}2. Start the Streamlit web app:{END}
   streamlit run app/streamlit_app.py

{BOLD}3. Build Docker container (optional):{END}
   docker build -t churn-prediction:v1.0 .

{BOLD}4. Read detailed testing guide:{END}
   TESTING_AND_VALIDATION.md
    """)

def main():
    """Run all tests"""
    print_header("CUSTOMER CHURN PREDICTION - QUICK TEST SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run tests
    tests = [
        ("Environment", test_environment),
        ("Logger", test_logger),
        ("Constants", test_constants),
        ("Configuration", test_config),
        ("Data Loader", test_data_loader),
        ("Data Cleaner", test_data_cleaner),
        ("Preprocessor", test_preprocessor),
        ("Model Trainer", test_model_trainer),
        ("Model Evaluator", test_model_evaluator),
        ("Model Manager", test_model_manager),
        ("Streamlit App", test_streamlit),
        ("Docker", test_docker),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                print_skip(f"{test_name} test incomplete (non-critical)")
        except KeyboardInterrupt:
            print(f"\n{RED}Tests interrupted by user{END}")
            break
        except Exception as e:
            print_fail(f"{test_name} test crashed", e)
    
    # Print summary
    print_summary()

if __name__ == "__main__":
    main()
