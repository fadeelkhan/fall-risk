# import necessary libraries
import pandas as pd
import numpy as np


def get_magnitude(a, b, c):
    return np.array([np.sqrt(a ** 2 + b ** 2 + c ** 2)])

def process_list(lst):
    '''
    Returns processed list with 78 synthesized features from a list with 9 features
    :param lst: a list with 9 features (acceleration, gyroscope and orientation in x, y, z directions)
    :return: numpy array with 78 synthesized features
    '''
    # Calculate Jerk
    # lst = np.array(list(map(float, lst.split(','))))
    max = lst.copy()
    min = lst.copy()
    dt = 150E-3  # 150ms
    avg_jerk = lst / dt
    acc_jerk_mag = get_magnitude(avg_jerk[0], avg_jerk[1], avg_jerk[2])
    gyro_jerk_mag = get_magnitude(avg_jerk[3], avg_jerk[4], avg_jerk[5])
    orient_jerk_mag = get_magnitude(avg_jerk[6], avg_jerk[7], avg_jerk[8])

    transformed_y = np.abs(np.fft.fft(lst))
    fmax = transformed_y.copy()
    fmin = transformed_y.copy()
    favg_jerk = np.abs(np.fft.fft(avg_jerk))
    f_acc_jerk_mag = get_magnitude(favg_jerk[0], favg_jerk[1], favg_jerk[2])
    f_gyro_jerk_mag = get_magnitude(favg_jerk[3], favg_jerk[4], favg_jerk[5])
    f_orient_jerk_mag = get_magnitude(favg_jerk[6], favg_jerk[7], favg_jerk[8])

    processed = np.concatenate((lst, max, min, avg_jerk, acc_jerk_mag, gyro_jerk_mag, orient_jerk_mag, transformed_y,
                                fmax, fmin, favg_jerk, f_acc_jerk_mag, f_gyro_jerk_mag, f_orient_jerk_mag))
    return processed

def get_processed_data(file):
    """
    Returns the processed dataframe from the raw csv file inputted
    :param file:  csv file with 10 columns (9 accelerometer data inputs, and 1 for times)
    :return: processed data frame with 78 synthesized features
    """
    df = pd.read_csv(file)
    df = pd.DataFrame(file, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll'])
    df = df.dropna(axis=0)
    df = df.loc[~(df == 0).all(axis=1)]
    times = df['times']
    df = df.drop(columns=['times'])
    df = df.dropna()

    # create list of all activity titles
    master = pd.DataFrame()
    inter = pd.DataFrame()
    freq_inter = pd.DataFrame()
    freqs_magnitude = pd.DataFrame()

    # This returns the fourier transform coefficients as complex numbers
    for idx in df.columns:
        transformed_y = np.fft.fft(df[idx])
        # Take the absolute value of the complex numbers for magnitude spectrum
        freqs_magnitude[idx] = np.abs(transformed_y)

    for index, row in df.iterrows():
        a = pd.DataFrame(row).transpose()
        inter = inter.append(df_maker(a, 'time'))
        # Apppending
        freq_inter = freq_inter.append(df_maker(a, 'freq'))

    prcoessed_df = pd.concat([inter, freq_inter], axis=1)

    return prcoessed_df

def df_maker(testdf, type_domain):
    means = pd.DataFrame()
    maxes = pd.DataFrame()
    mins = pd.DataFrame()
    Jerk = pd.DataFrame()

    # group all rows by label of activity
    test = testdf.copy()

    # add min and max values
    tester = testdf.copy()
    maxx = testdf.copy()
    minn = testdf.copy()

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
        avg_jerkk = tester.astype(float) / dt
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
        dt = 150  # 150ms
        avg_jerkk = tester.astype(float) / dt
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

    # # Moving labels from index to a column
    # means = means.reset_index(drop=False)
    # maxes = maxes.reset_index(drop=False)
    # mins = mins.reset_index(drop=False)
    # Jerk = Jerk.reset_index(drop=False)

    master = pd.concat([means, maxes, mins, Jerk], axis=1)
    return master