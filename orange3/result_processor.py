# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')
    

plot_RF = True


temp = "180"
path = "res_" + temp + ".csv"

df = pd.read_csv(path)
df = df.drop([0, 1])

x = np.array(df["x"]).astype(np.float32)
y = np.array(df["SVM (1)"]).astype(np.float16)
y_RF = np.array(df["Random Forest (1)"]).astype(np.float16)

plt.xlabel("Distance [micrometers]")
plt.ylabel("Normalized Peak Intensity [-]")
plt.plot(x, y, label="% PEI (SVM)", color="C1")
if plot_RF:
    plt.plot(x, y_RF, label="% PEI (RF)", color="C2")
plt.legend()
format_plot()

plt.show()

# Butterworth Filter
N = 3  # Filter order
Wn = 0.1  # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')
smooth_data = signal.filtfilt(B, A, y)
smooth_data_RF = signal.filtfilt(B, A, y_RF)

plt.xlabel("Distance [micrometers]")
plt.ylabel("Normalized Peak Intensity [-]")
plt.plot(x, smooth_data, label="% PEI (SVM)", color="C1")
if plot_RF:
    plt.plot(x, smooth_data_RF, label="% PEI (RF)", color="C2")
plt.legend()
format_plot()

plt.show()