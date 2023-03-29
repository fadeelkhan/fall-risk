'''
Used to make predictions based on already-trained data
'''
import pandas as pd
from pre_processing_functions.feature_engineering import get_processed_data
from prediction_functions.training import get_models

def predict_using_existing_models(data_file):
    binary_model, multiclass_model = get_models()
    binary_predictions, multiclass_predictions = predict(binary_model, multiclass_model, data_file)
    return binary_predictions, multiclass_predictions

def predict(binary_model, multiclass_model, data_file):
    processed_df = get_processed_data(data_file)
    binary_predictions = binary_model.predict(processed_df)
    multiclass_predictions = multiclass_model.predict(processed_df)

    return binary_predictions, multiclass_predictions

def get_times(data_file):
    df = pd.read_csv(data_file)
    return df['times']