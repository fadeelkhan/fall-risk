from prediction_functions import predictions
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# To re-train data on pre-processed training data
training_data = 'data/master_dataset.csv'
input_file = "data/Hour_Data_Stream.csv"


# Train and Predict
fall_vs_no_fall, types_of_activities = predictions.train_and_predict(training_data, input_file)
