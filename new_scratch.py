import PySimpleGUI as sg
from time import time
import random

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
                start_time = time()
                active = True
                window['-STARTSTOP-'].update('Stop')

    if active:
        activity_status = ['FALL', 'NON-FALL']
        window['-TIME-'].update(random.choice(activity_status))


window.close()