from flask import Blueprint, request, jsonify
import logging
import joblib
import pandas as pd
import sqlite3
from app.models.model_utils import load_model, predict_severity
from app.models.feature_engineering import transform_features, create_preprocessing_pipeline

bp = Blueprint('main', __name__)

# Define expected feature columns based on dataset
FEATURE_COLUMNS = [
    "Age_Band_of_Driver", "Age_of_Vehicle", "Engine_Capacity_.CC.", "Hit_Object_in_Carriageway",
    "Hit_Object_off_Carriageway", "Journey_Purpose_of_Driver", "Junction_Location", "Light_Conditions",
    "Number_of_Casualties", "Number_of_Vehicles", "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities", "Police_Force", "Road_Surface_Conditions", "Road_Type",
    "Speed_limit", "Time", "Urban_or_Rural_Area", "Weather_Conditions", "Year_y"
]

@bp.route('/predict', methods=['POST'])
def predict():
    logging.info("Received prediction request.")
    model = load_model()
    if not model:
        return jsonify({"error": "Model could not be loaded"}), 500

    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Ensure Accident_Index is retained for output
        accident_indices = df['Accident_Index'].copy()

        # Define the columns used in training, excluding 'Accident_Index' for transformations
        columns_used = [
            '1st_Road_Class', 'Day_of_Week', 'Junction_Detail', 'Light_Conditions', 'Number_of_Casualties',
            'Number_of_Vehicles', 'Road_Surface_Conditions', 'Road_Type', 'Special_Conditions_at_Site', 'Speed_limit',
            'Time', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Age_Band_of_Driver', 'Age_of_Vehicle',
            'Hit_Object_in_Carriageway', 'Hit_Object_off_Carriageway', 'make', 'Engine_Capacity_.CC.', 'Sex_of_Driver',
            'Skidding_and_Overturning', 'Vehicle_Manoeuvre', 'Vehicle_Type'
        ]

        # Select only the columns used in training
        df = df[columns_used]


        # Apply the loaded feature transformations
        preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.pkl')
        df_transformed = preprocessing_pipeline.transform(df)

        # Predict accident severity
        predictions = predict_severity(model, df_transformed)

        # Combine Accident_Index with predictions
        results = pd.DataFrame({
            'Accident_Index': accident_indices,
            'Prediction': predictions
        })

        # Ensure predictions are integers if they are not already
        results['Prediction'] = results['Prediction'].astype(int)

        # Map predictions to descriptive labels
        results['Prediction_Label'] = results['Prediction'].map({1: 'Slight', 0: 'Serious or Fatal'})

        logging.info("Prediction successful.")
        return jsonify(results.to_dict(orient='records'))
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction."}), 500

@bp.route('/log_model_performance', methods=['GET'])
def log_model_performance():
    """Endpoint to log model performance metrics for system administrators."""
    try:
        performance_metrics = joblib.load("model_performance_metrics.pkl")
        return jsonify(performance_metrics)
    except Exception as e:
        logging.error(f"Error retrieving performance metrics: {str(e)}")
        return jsonify({"error": "Could not retrieve performance metrics."}), 500

@bp.route('/upload_data', methods=['POST'])
def upload_data():
    """Endpoint for insurance analysts to upload accident data for batch processing."""
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Validate dataset columns
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing columns in dataset: {missing_features}"}), 400
        
        # Store data in SQLite database
        conn = sqlite3.connect("accidents.db")
        df.to_sql("accidents", conn, if_exists="append", index=False)
        conn.close()
        
        return jsonify({"message": "Data uploaded successfully."})
    except Exception as e:
        logging.error(f"Data upload error: {str(e)}")
        return jsonify({"error": "An error occurred while uploading data."}), 500

@bp.route('/report_accident', methods=['POST'])
def report_accident():
    """Endpoint for end users to report accidents."""
    try:
        data = request.get_json()
        
        # Validate input JSON structure
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected JSON object."}), 400
        
        df = pd.DataFrame([data])
        
        # Validate input fields
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        # Store accident report in SQLite database
        conn = sqlite3.connect("accidents.db")
        df.to_sql("accident_reports", conn, if_exists="append", index=False)
        conn.close()
        
        return jsonify({"message": "Accident report submitted successfully."})
    except Exception as e:
        logging.error(f"Accident reporting error: {str(e)}")
        return jsonify({"error": "An error occurred while submitting the accident report."}), 500 