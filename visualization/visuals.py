import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
from random import randint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure
from visualization.mapping import preprocess_mapping
from visualization.mapping import create_heatmap
import seaborn as sns

def GUI(times, fall_vs_no_fall_predictions, location_file, image_file):
    layout = [[sg.Button('Plot Falls'), sg.Button('Plot Location Mapping'), sg.Button('Show Tabular Fall Data'), sg.Button('Export Data'), sg.Cancel()],
              [sg.Canvas(size=(20, 20), key='-CANVAS-')]]
    window = sg.Window('Physician Interface', layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event == 'Plot Falls':
            draw_plot(times, fall_vs_no_fall_predictions, window)
        elif event == 'Plot Location Mapping':
            location(location_file, image_file, window)
        elif event == 'Show Tabular Fall Data':
            make_table(times, fall_vs_no_fall_predictions, location_file, window)
    window.close()

def draw_plot(times, fall_vs_no_fall_predictions, window):
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time")
    ax.set_ylabel("No Fall [0] vs Fall [1]")
    ax.plot(times, fall_vs_no_fall_predictions)
    fig_agg = draw_figure(canvas, fig)
    fig_agg.draw()

def location(location_file, image_file, window):
    df = preprocess_mapping(location_file)
    df, df2 = create_heatmap(df, image_file)
    di = {107: 'k', 114: 'r'}
    df2 = df2.replace({"color": di})

    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time")
    ax.set_ylabel("No Fall [0] vs Fall [1]")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df2['x'], df2['y'], s=1, color=df2['color']) # plotting location

    sns.kdeplot(data=df, x="x1", y="y1", fill=True, alpha=0.5) # plotting heatmap

    fig_agg = draw_figure(canvas, fig)
    fig_agg.draw()


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def plot_falls_and_activities(times, fall_vs_no_fall_predictions):
    plt.plot(times, fall_vs_no_fall_predictions)
    plt.xlabel('Time')
    plt.ylabel('Fall (1) vs. No Fall (0)')
    plt.show()

def make_table(times, fall_vs_no_fall_predictions, location_file, window):
    df = preprocess_mapping(location_file)
    df = df.dropna()
    df = df.head(len(fall_vs_no_fall_predictions))
    df['Fall Status'] = fall_vs_no_fall_predictions
    # df['Time of Event'] = times
    # print(df)
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    fig = Figure()
    ax = fig.add_subplot(111)
    table = ax.table(cellText=df.values, colLabels=df.columns)
    fig.tight_layout()
    table.set_fontsize(14)
    table.scale(1, 4)
    ax.axis('off')
    fig_agg = draw_figure(canvas, fig)
    fig_agg.draw()

