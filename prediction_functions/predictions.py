'''
Used to make predictions based on already-trained data
'''
import pandas as pd
import numpy as np
from prediction_functions import training
from pre_processing_functions import pre_processing

def train_and_predict(training_data, x_test_data):
    binary_model, multiclass_model = training.get_trained_models(training_data)
    processed_df = pre_processing.get_processed_data(x_test_data)

    binary_predictions = binary_model.predict(processed_df)
    multiclass_predictions = multiclass_model.predict(processed_df)

    return binary_predictions, multiclass_predictions
