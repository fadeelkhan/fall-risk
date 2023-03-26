from prediction_functions import predictions
from pre_processing_functions import pre_processing_raw_data
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# create master dataset for training data
# Location of raw data
# location = '/content/drive/Shareddrives/Team Three Seasons/Cycle 3/fall_data/annotated_data'
# pre_processing_raw_data.create_master_dataframe(location)

# To re-train data on pre-processed training data
training_data = 'data/master_dataset.csv'
input_file = "data/Hour_Data_Stream.csv"


# Train and Predict
fall_vs_no_fall_predictions, types_of_activities_predictions = predictions.train_and_predict(training_data, input_file)
