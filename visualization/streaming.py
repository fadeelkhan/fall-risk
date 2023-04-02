import serial.tools.list_ports
import regex as re
from prediction_functions import predictions

def stream():
    ports = serial.tools.list_ports.comports()
    serialInst = serial.Serial()

    portsList = []

    for onePort in ports:
        portsList.append(str(onePort))

    val = 3

    for x in range(0,len(portsList)):
        if portsList[x].startswith("COM" + str(val)):
            portVar = "COM" + str(val)

    serialInst.baudrate = 9600
    serialInst.port = portVar
    serialInst.open()



    while True:
        if serialInst.in_waiting:
            packet = serialInst.readline()
            input_file = packet.decode('utf').rstrip('\n')
            if bool(re.search(r'\d', input_file)):
                # input_file = "data/Hour_Data_Stream.csv"

                # # # Train and Predict
                fall_vs_no_fall_predictions, types_of_activities_predictions = predictions.predict_using_existing_models(input_file.rstrip())
                # times = predictions.get_times(input_file)

                # # Make visualizations and GUI
                # visuals.GUI(times[:6], fall_vs_no_fall_predictions[:6], location_file, mapped_location)
                # visuals.GUI(times, fall_vs_no_fall_predictions, location_file, mapped_location)
                return types_of_activities_predictions
