from prediction_functions import predictions
from pre_processing_functions import pre_processing_raw_data
from prediction_functions.training import save_new_trained_models
from visualization import visuals
import warnings
warnings.filterwarnings("ignore")

## PREPROCESSING OF RAW CLINICAL TRIAL DATA
# Create a master processed dataset from annotated raw data from clinical trials:

# Location of raw data from clinical trials. Each file should have 9 features and a label:
# location = '/content/drive/Shareddrives/Team Three Seasons/Cycle 3/fall_data/annotated_data'

# Create dataset:
# pre_processing_raw_data.create_master_dataframe(location)


## TRAINING AND PREDICTION
# # To re-train data on pre-processed training data
# training_data = 'data/master_dataset.csv'
# save_new_trained_models(training_data)

input_file = "data/Hour_Data_Stream.csv"

# # Train and Predict
fall_vs_no_fall_predictions, types_of_activities_predictions = predictions.predict_using_existing_models(input_file)
times = predictions.get_times(input_file)

# Make visualizations and GUI
visuals.plot_falls_and_activities(times, fall_vs_no_fall_predictions)
