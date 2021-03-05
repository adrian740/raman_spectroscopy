import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from os import listdir
from os.path import isfile, join

path = "data analysis project//120oC//"

allfiles = [f for f in listdir(path) if isfile(join(path, f))]

def read_data(filename):
    data = np.genfromtxt(filename, delimiter="\t")

    range = data[:, 0]
    shift = data[:, 1]

    N = 3  # Filter order
    Wn = 0.1  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B, A, shift)

    return range, shift, smooth_data

i = 0
skip = 20

for filename in allfiles:
    if filename.endswith(".txt"):
        i += 1
        if i % skip is not 0:
            continue
        range_, raw_data, smooth_data = read_data(path + filename)
        plt.plot(range_, smooth_data, label=filename)

plt.legend()
plt.show()