from src.models.model_manager import ModelManager

model_path = 'models/churn_model_best.pkl'
model, scaler, preprocessor, metadata = ModelManager.load_model(model_path)
print('Model class:', type(model).__name__)
print('Metadata model_type:', metadata.get('model_type'))
print('Metadata metrics ROC-AUC:', metadata.get('metrics', {}).get('ROC-AUC'))
