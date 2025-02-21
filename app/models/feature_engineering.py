import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, KBinsDiscretizer, MaxAbsScaler
from sklearn.impute import SimpleImputer
import datetime  # Import the datetime module
from sklearn.compose import ColumnTransformer

def get_Speed_limit(df):
    return df[['Speed_limit']]

def get_Time(df):
    return pd.to_datetime(df['Time'], format='%H:%M').dt.time

def find_time_group(time_object):
    if pd.isna(time_object):
        return 'Unknown'  # Return 'Unknown' or another appropriate category for missing times
    if time_object < datetime.time(5, 0):
        return 'Night'
    elif time_object < datetime.time(7, 0):
        return 'Early Morning'
    elif time_object < datetime.time(10, 0):
        return 'Morning'
    elif time_object < datetime.time(15, 0):
        return 'Midday'
    elif time_object < datetime.time(18, 0):
        return 'Afternoon'
    elif time_object < datetime.time(20, 0):
        return 'Evening'
    elif time_object <= datetime.time(23, 59):
        return 'Late Evening'
    return 'Unknown'

def get_Age_of_Vehicle(df):
    return df[['Age_of_Vehicle']]

def get_make(df):
    list_of_small_makers = list(df['make'].value_counts()[df['make'].value_counts() < 2000].index)
    return df['make'].replace(list_of_small_makers, 'Other').to_frame()

def get_Engine_Capacity(df):
    return df[['Engine_Capacity_.CC.']]

def get_columns_to_one_hot(df):
    return df[['1st_Road_Class', 'Day_of_Week', 'Junction_Detail', 'Light_Conditions', 'Number_of_Casualties', 
               'Number_of_Vehicles', 'Road_Surface_Conditions', 'Road_Type', 'Special_Conditions_at_Site', 
               'Urban_or_Rural_Area', 'Weather_Conditions', 'Age_Band_of_Driver', 'Hit_Object_in_Carriageway',
               'Hit_Object_off_Carriageway', 'Sex_of_Driver', 'Skidding_and_Overturning',
               'Vehicle_Manoeuvre', 'Vehicle_Type']]

# Define all your pipelines here
FullTransformerOnMake = Pipeline([("Select_Make",      FunctionTransformer(func=get_make, validate=False)),
                                   ("Fill_Null",       SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Other')),
                                   ("One_Hot_Encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])

# Add other transformers similarly
FullTransformerOnEngineCapacity = Pipeline([("Select_Engine_Capacity",       FunctionTransformer(func=get_Engine_Capacity, validate=False)),
                                            ("Fill_Null",                    SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                                            ("Car_Types_by_Engine_Capacity", KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='quantile')),
                                            ("One_Hot_Encoder",              OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                                           ])

DataToOneHotTransformerOnColumns = Pipeline([
    ("Select_Columns",  FunctionTransformer(func=get_columns_to_one_hot, validate=False)),
    ("One_Hot_Encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

FullTransformerOnAgeofVehicle = Pipeline([("Select_Age_of_Vehicle", FunctionTransformer(func=get_Age_of_Vehicle, validate=False)),
                                          ("Fill_Null",             SimpleImputer(missing_values=np.nan, strategy='median'))
                                         ])

def time_grouping(x):
    return x.apply(find_time_group).to_frame()

FullTransformerOnTime = Pipeline([
    ("Select_Time", FunctionTransformer(func=get_Time, validate=False)),
    ("Group_Time", FunctionTransformer(func=time_grouping, validate=False)),
    ("Fill_Null", SimpleImputer(missing_values=np.nan, strategy='most_frequent', fill_value='Unknown')),
    ("One_Hot_Encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

FullTransformerOnSpeedLimit = Pipeline([("Select_Speed_Limit", FunctionTransformer(func=get_Speed_limit, validate=False)),
                                        ("Fill_Null",          SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                                        ("One_Hot_Encoder",    OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                                       ])

# Correctly define FeatureUnion with named transformers
FeatureUnionTransformer = FeatureUnion([
    ("FTAgeofVehicle", FullTransformerOnAgeofVehicle),
    ("FTEngineCapacity", FullTransformerOnEngineCapacity),
    ("FTMake", FullTransformerOnMake),
    ("FTSpeedLimit", FullTransformerOnSpeedLimit),
    ("FTTime", FullTransformerOnTime),
    ("OHEColumns", DataToOneHotTransformerOnColumns)
])

# Full feature engineering pipeline
Full_Transformer = Pipeline([
    ("Feature_Engineering", FeatureUnionTransformer),
    ("Min_Max_Transformer", MaxAbsScaler())
])

def transform_features(X):
    return Full_Transformer.transform(X)

def create_preprocessing_pipeline():
    # Define the preprocessing steps
    preprocessing_steps = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), ['1st_Road_Class', 'Day_of_Week', 'Junction_Detail', 'Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', 'Special_Conditions_at_Site', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Sex_of_Driver', 'Skidding_and_Overturning', 'Vehicle_Manoeuvre', 'Vehicle_Type']),
            # Add other transformers as needed
        ])

    # Create a pipeline that applies these steps
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessing_steps),
        ('onehot_encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    return pipeline