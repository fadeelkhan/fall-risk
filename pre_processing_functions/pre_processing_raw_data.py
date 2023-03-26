import pandas as pd
import os
import glob
import numpy as np
from pathlib import Path

def create_master_dataframe(location):
    # use glob to get all the csv files
    # in the folder
    path = os.getcwd()
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    standinglist = (os.listdir(location))

    # create list of all activity titles
    master = pd.DataFrame()
    inter = pd.DataFrame()
    freq_inter = pd.DataFrame()

    for i in standinglist:
        freqs_magnitude = pd.DataFrame()
        filepath = location + '/' + i
        # convert file in testdf
        main_df = pd.read_csv(filepath)
        # Frequency Domain
        # This returns the fourier transform coeficients as complex numbers
        features = main_df.drop(columns=['label'])
        for idx in features.columns:
            transformed_y = np.fft.fft(main_df[idx])
            # Take the absolute value of the complex numbers for magnitude spectrum
            freqs_magnitude[idx] = np.abs(transformed_y)

        freqs_magnitude['label'] = main_df['label']

        # Apppending
        inter = inter.append(df_maker(main_df, 'time'))
        freq_inter = freq_inter.append(df_maker(freqs_magnitude, 'freq'))
        freq_inter = freq_inter.drop(columns=['label'])

    master = pd.concat([inter, freq_inter], axis=1)

    filepath = Path(os.path.abspath(os.curdir))
    filepath = filepath.parent()
    filepath = filepath + "/data/master_dataset.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(filepath)


def df_maker(testdf, type_domain):
    means = pd.DataFrame()
    maxes = pd.DataFrame()
    mins = pd.DataFrame()
    Jerk = pd.DataFrame()

    # group all rows by label of activity
    test = testdf.groupby(['label'])

    # add min and max values
    tester = test.mean()
    maxx = testdf.groupby(['label']).max()
    minn = testdf.groupby(['label']).min()

    # Renaming
    if type_domain == 'freq':
        tester = tester.rename(
            columns={"acc_x": "F_acc_x", "acc_y": "F_acc_y", "acc_z": "F_acc_z", "gyro_x": "F_gyro_x",
                     "gyro_y": "F_gyro_y", "gyro_z": "F_gyro_z", "azimuth": "F_azimuth", "pitch": "F_pitch",
                     "roll": "F_roll"})
        maxx = maxx.rename(
            columns={"acc_x": "Fmax_acc_x", "acc_y": "Fmax_acc_y", "acc_z": "Fmax_acc_z", "gyro_x": "Fmax_gyro_x",
                     "gyro_y": "Fmax_gyro_y", "gyro_z": "Fmax_gyro_z", "azimuth": "Fmax_azimuth", "pitch": "Fmax_pitch",
                     "roll": "Fmax_roll"})
        minn = minn.rename(
            columns={"acc_x": "Fmin_acc_x", "acc_y": "Fmin_acc_y", "acc_z": "Fmin_acc_z", "gyro_x": "Fmin_gyro_x",
                     "gyro_y": "Fmin_gyro_y", "gyro_z": "Fmin_gyro_z", "azimuth": "Fmin_azimuth", "pitch": "Fmin_pitch",
                     "roll": "Fmin_roll"})

        # Jerk
        dt = 150E-3  # 150ms
        avg_jerkk = tester / dt
        avg_jerkk = avg_jerkk.rename(
            columns={"F_acc_x": "jerk_Facc_x", "F_acc_y": "jerk_Facc_y", "F_acc_z": "jerk_Facc_z",
                     "F_gyro_x": "jerk_Fgyro_x", "F_gyro_y": "jerk_Fgyro_y", "F_gyro_z": "jerk_Fgyro_z",
                     "F_azimuth": "jerk_Fazimuth", "F_pitch": "jerk_Fpitch", "F_roll": "jerk_Froll"})
        avg_jerkk['F_JerkMagAcc'] = np.sqrt(
            (avg_jerkk['jerk_Facc_x']) ** 2 + (avg_jerkk['jerk_Facc_y']) ** 2 + (avg_jerkk['jerk_Facc_z']) ** 2)
        avg_jerkk['F_JerkMagGyro'] = np.sqrt(
            (avg_jerkk['jerk_Fgyro_x']) ** 2 + (avg_jerkk['jerk_Fgyro_y']) ** 2 + (avg_jerkk['jerk_Fgyro_z']) ** 2)
        avg_jerkk['F_JerkMagOrient'] = np.sqrt(
            (avg_jerkk['jerk_Fazimuth']) ** 2 + (avg_jerkk['jerk_Fpitch']) ** 2 + (avg_jerkk['jerk_Froll']) ** 2)

    else:
        maxx = maxx.rename(
            columns={"acc_x": "max_acc_x", "acc_y": "max_acc_y", "acc_z": "max_acc_z", "gyro_x": "max_gyro_x",
                     "gyro_y": "max_gyro_y", "gyro_z": "max_gyro_z", "azimuth": "max_azimuth", "pitch": "max_pitch",
                     "roll": "max_roll"})
        minn = minn.rename(
            columns={"acc_x": "min_acc_x", "acc_y": "min_acc_y", "acc_z": "min_acc_z", "gyro_x": "min_gyro_x",
                     "gyro_y": "min_gyro_y", "gyro_z": "min_gyro_z", "azimuth": "min_azimuth", "pitch": "min_pitch",
                     "roll": "min_roll"})

        # Jerk
        dt = 150E-3  # 150ms
        avg_jerkk = tester / dt
        avg_jerkk = avg_jerkk.rename(
            columns={"acc_x": "jerk_acc_x", "acc_y": "jerk_acc_y", "acc_z": "jerk_acc_z", "gyro_x": "jerk_gyro_x",
                     "gyro_y": "jerk_gyro_y", "gyro_z": "jerk_gyro_z", "azimuth": "jerk_azimuth", "pitch": "jerk_pitch",
                     "roll": "jerk_roll"})
        avg_jerkk['JerkMagAcc'] = np.sqrt(
            (avg_jerkk['jerk_acc_x']) ** 2 + (avg_jerkk['jerk_acc_y']) ** 2 + (avg_jerkk['jerk_acc_z']) ** 2)
        avg_jerkk['JerkMagGyro'] = np.sqrt(
            (avg_jerkk['jerk_gyro_x']) ** 2 + (avg_jerkk['jerk_gyro_y']) ** 2 + (avg_jerkk['jerk_gyro_z']) ** 2)
        avg_jerkk['JerkMagOrient'] = np.sqrt(
            (avg_jerkk['jerk_azimuth']) ** 2 + (avg_jerkk['jerk_pitch']) ** 2 + (avg_jerkk['jerk_roll']) ** 2)

    # add aggregated data into master data frame
    means = means.append(tester)
    mins = mins.append(minn)
    maxes = maxes.append(maxx)
    Jerk = Jerk.append(avg_jerkk)

    # Moving labels from index to a column
    means = means.reset_index(drop=False)
    maxes = maxes.reset_index(drop=False)
    mins = mins.reset_index(drop=False)
    Jerk = Jerk.reset_index(drop=False)

    maxes = maxes.drop(columns=['label'])
    mins = mins.drop(columns=['label'])
    Jerk = Jerk.drop(columns=['label'])

    master = pd.concat([means, maxes, mins, Jerk], axis=1)
    return master