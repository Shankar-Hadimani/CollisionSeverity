import joblib
import os

# Define the path where the model is stored
model_path = os.path.join('models', 'accident_severity_model.pkl')

def load_model():
    """Load the trained model from the specified path."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return None

def predict_severity(model, features):
    """Predict the severity given a model and features."""
    return model.predict(features) 