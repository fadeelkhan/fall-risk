from pre_processing_functions import pre_processing_raw_data
from prediction_functions.training import save_new_trained_models
from visualization import visuals
import warnings
import PySimpleGUI as sg
from visualization.streaming import stream


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

# location_file = 'visualization/UWB_mappingtest1.xlsx'
# mapped_location = 'visualization/Breakroom.csv'
###

# fall_vs_no_fall_predictions, types_of_activities_predictions = predictions.predict_using_existing_models(input_file)
# times = predictions.get_times(input_file)

# # Make visualizations and GUI
# visuals.GUI(times[:6], fall_vs_no_fall_predictions[:6], location_file, mapped_location)
# visuals.GUI(times, fall_vs_no_fall_predictions, location_file, mapped_location)


def create_window():
    sg.theme('black')
    layout = [
        [sg.Push(), sg.Image('cross.png', pad=0, enable_events=True, key='-CLOSE-')],
        [sg.VPush()],
        [sg.Text('', font='Young 50', key='-TIME-')],
        [
            sg.Button('Start', button_color=('#FFFFFF', '#FF0000'), border_width=0, key='-STARTSTOP-'),
        ],
        [sg.VPush()]
    ]

    return sg.Window(
        'Physician Interface',
        layout,
        size=(600, 600),
        no_titlebar=True,
        element_justification='center')


window = create_window()
start_time = 0
active = False
lap_amount = 1

while True:
    event, values = window.read(timeout=10)
    if event in (sg.WIN_CLOSED, '-CLOSE-'):
        break

    if event == '-STARTSTOP-':
        if active:
            # from active to stop
            active = False
            window['-STARTSTOP-'].update('Reset')
        else:
            # from stop to reset
            if start_time > 0:
                window.close()
                window = create_window()
                start_time = 0
                lap_amount = 1
            # from start to active
            else:
                active = True
                window['-STARTSTOP-'].update('Stop')

    if active:
        activity_status = stream()
        window['-TIME-'].update(activity_status[0])


window.close()
