'''
Used to make predictions based on already-trained data
'''
import pandas as pd
from pre_processing_functions.feature_engineering import get_processed_data
from prediction_functions.training import get_models
from pre_processing_functions.feature_engineering import process_list

def real_time_predictions():
    pass

def predict_using_existing_models(data_file):
    binary_model, multiclass_model = get_models()
    binary_predictions, multiclass_predictions = predict(binary_model, multiclass_model, data_file)

    binary_mapping = {0: 'No-Fall', 1: "Fall"}
    binary_predictions = list(map(binary_mapping.get, binary_predictions))

    # multi_mapping = {'A1': 'Walking', 'A2': 'Walking Quickly', 'A3': 'Walking Upstairs Slowly', 'A4': 'Walking Upstairs Quickly',
    #                  'A5': 'Slowly Sit In A Chair, Then Get Up', 'A6': 'Quickly Sit In A Chair, Then Get Up', 'A7': 'Slowly Sit In A Low Height Chair, Then Get Up',
    #                  'A8': 'Quicly Sit In A Low Height Chair, Then Get Up', 'A9': 'Sitting In A Chair, Trying To Get Up and Collapsing',
    #                  'A12': 'Standing, Slowly Bending At Knees, Then Getting Up', 'A13': 'Going Into A Deadlift Position', 'A14': 'Stumble While Walking',
    #                  'A15': 'Trying To Reach A High Place', 'A16': 'Walking Downstairs Slowly', 'A17': 'Walking Downstairs Quickly', 'F1': 'Falling Forward While Walking After Slipping',
    #                  'F2': 'Falling Backward While Walking After Slipping', 'F3': 'Lateral Fall While Walking After Slipping', 'F4': 'Falling Forward While Walking After Tripping',
    #                  'F5': 'Falling While Walking, Using Hands to Dampen Fall', 'F6': 'Fall Forward When Trying To Get Up'}

    multi_mapping = {'F': 1, 'S': 0, 'T': 0, 'W': 0}

    multiclass_predictions = list(map(multi_mapping.get, multiclass_predictions))
    return binary_predictions, multiclass_predictions

def predict(binary_model, multiclass_model, data_file):
    processed_df = get_processed_data(data_file)
    # processed_df = process_list(data_file)
    # binary_predictions = binary_model.predict(processed_df.reshape(1, -1))
    binary_predictions = binary_model.predict(processed_df)

    # multiclass_predictions = multiclass_model.predict(processed_df.reshape(1, -1))
    multiclass_predictions = multiclass_model.predict(processed_df)

    return binary_predictions, multiclass_predictions

def get_times(data_file):
    # df = pd.read_csv(data_file)
    return data_file['times']

def tabulate_fall_coordinates(falls, locations):

    return 1
