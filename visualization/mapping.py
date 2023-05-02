import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def preprocess_mapping(data_file):
    df = pd.read_excel(data_file)

    # filter out rows where there is only an integer
    # df = df[df.apply(lambda x: x.str.isnumeric().all(), axis=1)]

    df = df.iloc[:, :2]
    df.columns = ['x', 'y']
    return df

def create_heatmap(df, mapped_location_file):
    q1 = df["x"].quantile(0.95)
    q2 = df["y"].quantile(0.95)
    df = df[df["x"] < q1]
    df = df[df["y"] < q2]

    df2 = pd.read_csv(mapped_location_file)
    df2 = df2[(df2 > -1).all(1)]

    # ax.scatter(actual_x,actual_y, s = 8, color='g') # plotting location
    # circle1 = plt.Circle((actual_x[0],actual_y[0]),0.91,color='g', fill=False)
    # ax.add_patch(circle1)
    # circle2 = plt.Circle((actual_x[1],actual_y[1]),0.91,color='g', fill=False)
    # ax.add_patch(circle2)
    # circle3 = plt.Circle((actual_x[2],actual_y[2]),0.91,color='g', fill=False)
    # ax.add_patch(circle3)
    # circle4 = plt.Circle((actual_x[3],actual_y[3]),0.91,color='g', fill=False)
    # ax.add_patch(circle4)

    # plt.title('OEDK Breakroom + Countour Map of Time Spent')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.show()

    return df, df2

