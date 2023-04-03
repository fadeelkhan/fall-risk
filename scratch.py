import serial.tools.list_ports
import numpy as np
import regex as re

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial('COM3', 9600, timeout=100)
# serialInst.open()
n=0
while True:
    packet = serialInst.readline()
    input_file = packet.decode('utf', errors='ignore').rstrip('\n')
    input_file = np.array(list(map(float, re.findall('[-+]?[0-9]*\.?[0-9]+', input_file))))
    while True:
        n+=1
        new_packet = serialInst.readline()
        new_lst = new_packet.decode('utf', errors='ignore').rstrip('\n')
        new_lst = np.array(list(map(float, re.findall('[-+]?[0-9]*\.?[0-9]+', new_lst))))
        print(n, new_lst)
        if n//10 ==0:
            # input_file = (input_file + new_lst) / 2
            break

