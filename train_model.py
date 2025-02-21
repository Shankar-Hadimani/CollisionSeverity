import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from app.models.feature_engineering import Full_Transformer, transform_features
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Load data
    logging.info("Loading data...")
    df = pd.read_csv(os.path.join('data', 'final_accident_vehicle_dataset.csv'))
    logging.info("Data loaded successfully.")
    df = df[['Accident_Index', '1st_Road_Class','Day_of_Week', 'Junction_Detail','Light_Conditions', 'Number_of_Casualties',
              'Number_of_Vehicles', 'Road_Surface_Conditions', 'Road_Type', 'Special_Conditions_at_Site', 'Speed_limit',
              'Time', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Age_Band_of_Driver', 'Age_of_Vehicle',
              'Hit_Object_in_Carriageway', 'Hit_Object_off_Carriageway', 'make', 'Engine_Capacity_.CC.', 'Sex_of_Driver',
              'Skidding_and_Overturning', 'Vehicle_Manoeuvre', 'Vehicle_Type', 'Accident_Severity'
             ]]
    df['Accident_Severity'] = df['Accident_Severity'].replace(['Serious', 'Fatal'], 'Serious or Fatal')
    df = pd.get_dummies(df, columns=['Accident_Severity'])
    df = df.drop('Accident_Severity_Serious or Fatal', axis=1)


    # Preprocess and feature engineering
    logging.info("Applying feature transformations...")
    X = df.drop(['Accident_Index', 'Accident_Severity_Slight'], axis=1)
    y = df['Accident_Severity_Slight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Full_Transformer.fit(X_train)
    X_train_transformed = Full_Transformer.transform(X_train)
    X_test_transformed = Full_Transformer.transform(X_test)
    logging.info("Feature transformations applied.")

    # Train model
    logging.info("Training model...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=3)
    clf.fit(X_train_transformed, y_train)
    logging.info("Model trained successfully.")

    # Predictions and evaluations
    y_pred = clf.predict(X_test_transformed)
    y_pred_proba = clf.predict_proba(X_test_transformed)[:, 1]
    logging.info("Predictions made.")

    # Create directories for saving plots and metrics
    os.makedirs('plots', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    # Classification Report
    report = classification_report(y_test, y_pred)
    with open('metrics/classification_report.txt', 'w') as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()

    # ROC AUC Score
    with open('metrics/roc_auc_score.txt', 'w') as f:
        f.write(f'ROC AUC Score: {roc_auc}\n')

    # Save model
    model_save_path = os.path.join('models', 'accident_severity_model.pkl')
    joblib.dump(clf, model_save_path)
    logging.info(f"Model saved at {model_save_path}.")

    # Save the fitted pipeline
    pipeline_save_path = os.path.join('models', 'preprocessing_pipeline.pkl')
    joblib.dump(Full_Transformer, pipeline_save_path)
    logging.info(f"Pipeline saved at {pipeline_save_path}.")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")