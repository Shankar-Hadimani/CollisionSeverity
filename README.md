# Accident Severity Prediction System

## Overview
This project aims to develop an **Accident Severity Prediction System** using machine learning and a Flask-based API. The system leverages historical accident data to predict the severity of new accidents based on user input. The target audience includes **insurance analysts**, **system administrators**, **data scientists**, and **end users (accident reporters)**.

## Business Problem
Car insurance companies struggle to accurately assess accident severity, impacting risk evaluation and premium pricing. Leveraging UK road safety data, predictive models can enhance risk assessment by identifying key factors influencing accident outcomes. The system provides real-time predictions and structured analytics to improve decision-making.

## Key Features
- **Accident Severity Prediction:** Predicts accident severity (Slight, Serious, Fatal) based on input features.
- **Batch Data Upload:** Insurance analysts can upload accident datasets for analysis.
- **Model Training & Evaluation:** System administrators and data scientists can train, optimize, and evaluate models.
- **Accident Reporting:** End users can report accidents and receive severity predictions.
- **Model Performance Logging:** Logs accuracy, precision, recall, and F1-score.

## Tech Stack
- **Backend:** Flask, SQLite
- **Machine Learning:** Scikit-Learn, RandomForestClassifier
- **Data Processing:** Pandas, Joblib

## Dataset
The dataset used for training and evaluation consists of accident records, including:
- **Demographic Features:** Age Band of Driver, Engine Capacity, etc.
- **Environmental Features:** Light Conditions, Weather Conditions, Road Surface Conditions.
- **Incident Features:** Number of Vehicles, Pedestrian Crossings, Road Type, Speed Limit.
- **Target Variable:** Severity (Slight, Serious, Fatal)

## User Stories
### **1. Insurance Analyst**
- **Upload accident datasets for batch processing.**
- **Retrieve insights on past accident data and trends.**

### **2. System Administrator**
- **Monitor and log model performance metrics.**
- **Manually retrain models and manage data pipelines.**

### **3. Data Scientist**
- **Train, evaluate, and optimize the ML model.**
- **Perform hyperparameter tuning to improve accuracy.**
- **Monitor and analyze prediction errors.**

### **4. End User (Accident Reporter)**
- **Submit accident details and receive severity predictions.**
- **View reported accidents and predictions.**

## API Endpoints
### **1. Predict Accident Severity**
`POST /predict`
- **Input:** JSON with accident details.
- **Output:** Predicted severity level.

### **2. Upload Dataset**
`POST /upload_data`
- **Input:** CSV file containing accident records.
- **Output:** Confirmation of successful upload.

### **3. Train Model**
`POST /train_model`
- **Input:** Uses existing dataset stored in SQLite.
- **Output:** Trained model and performance metrics.

### **4. Retrieve Model Performance**
`GET /log_model_performance`
- **Output:** JSON with accuracy, precision, recall, and F1-score.

### **5. Report an Accident**
`POST /report_accident`
- **Input:** JSON with accident details.
- **Output:** Confirmation message.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourrepo/accident-severity-prediction.git
   cd accident-severity-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Train the model:
   ```bash
   python train_model.py
   ```
4. Run Prediction:
   ```bash
   python test_prediction.py
   ```

## Test Cases
### **1. Positive Case**
- **Input:** Valid accident details.
- **Expected Output:** Correct severity prediction.

### **2. Negative Case**
- **Input:** Missing or invalid accident details.
- **Expected Output:** Error message specifying missing fields.

### **3. Edge Case**
- **Input:** Extreme values (e.g., 100+ vehicles, unknown road conditions).
- **Expected Output:** Proper classification or flagged for manual review.

### **4. Model Performance Evaluation**
- **Input:** Performance metrics request.
- **Expected Output:** Accuracy, precision, recall, and F1-score.

## Future Enhancements
- **Real-time data ingestion using APIs.**
- **Integration with external road safety databases.**
- **Improved UI for accident reporting and analysis.**




