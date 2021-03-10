import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from os import listdir
from os.path import isfile, join

# Path containing the Raman Analysis
path = "data analysis project//120oC//"

# All of the files in the directory (only analyze .txt files)
allfiles = [f for f in listdir(path) if isfile(join(path, f))]

# Read the data from the file
def read_data(filename):
    data = np.genfromtxt(filename, delimiter="\t")

    shift = data[:, 0]
    intensity = data[:, 1]

    # Butterworth Filter
    N = 3  # Filter order
    Wn = 0.1  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B, A, intensity)

    # raman shift, amplitude, + smoothed amplitude
    return shift, intensity, smooth_data, float(filename.split("_")[-7])

#
# SETTINGS
#

# Plot settings
plot_ = False

# How many files to skip to plot one
skip = 50
idx = 0

# Dictionaries containing data
mapping = {}
data_dict = {}

# Raman Analysis Settings
#   PEAK: Main peak location
#   WIDTH: width of max search
pei_peak = 1004
pei_width = 20

epoxy_peak = 987
epoxy_width = 20

# Read in data
for filename in allfiles:
    if filename.endswith(".txt"):
        mapping[filename.split("_")[4]] = filename
ordered_keys = sorted(mapping, key = lambda x: float(x))

# Actually plot (+ process data)
for i in ordered_keys:
    filename = mapping[i]
    shift, intensity, smooth_intensity_data, x_coord = read_data(path + filename)
    idx += 1
    if plot_ and idx % skip is 0:
        plt.plot(shift, smooth_intensity_data, label=filename)
    data_dict[i] = [shift, intensity, smooth_intensity_data, x_coord]

if plot_:
    plt.legend()
    plt.show()

# Find the nearest value to the array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# Obtain Raman Max in range
pei_max = []
epo_max = []
x_val = []

for i in ordered_keys:
    shift, intensity, smooth_intensity_data, x_coord = data_dict.get(i)

    # PEI Analysis
    pei_idx_low, piw_val_low = find_nearest(shift, pei_peak - pei_width / 2)
    pei_idx_high, piw_val_high = find_nearest(shift, pei_peak + pei_width / 2)

    pei_dat = intensity[pei_idx_high:pei_idx_low]
    pei_max.append(max(pei_dat))

    # EPOXY Analysis
    epo_idx_low, piw_epo_low = find_nearest(shift, epoxy_peak - epoxy_width / 2)
    epo_idx_high, piw_epo_high = find_nearest(shift, epoxy_peak + epoxy_width / 2)

    epo_dat = intensity[epo_idx_high:epo_idx_low]
    epo_max.append(max(epo_dat))

    # x value
    x_val.append(x_coord)

# Data is read from back to front, so reverse
x_val = np.flip(x_val) # See below for my argument for the inclusion of this line
pei_max = np.flip(pei_max)
epo_max = np.flip(epo_max)

# Plot the concentration profiles
plt.plot(x_val, epo_max, label="EPOXY")
plt.plot(x_val, pei_max, label="PEI")
plt.legend()
plt.show()

# Remaining Questions:
#   Why do they reach similar maximum value (ie same order of magnitude)
#   Why the selected value for cm^-1? - use papers, specifically TUDelft Gradient Tg
#   How to normalize it? - ie values in example are not exactly between 0 and 1
#   Process + make the graph better

# Test plot the concentration
for i in ordered_keys:
    shift, intensity, smooth_intensity_data, x_coord = data_dict.get(i)

    if abs(x_coord + 40.0) < 0.01 or abs(x_coord - 70.0) < 0.01:
        plt.plot(shift, intensity, label=("x = " + str(x_coord)))

plt.plot([pei_peak, pei_peak], [0, 5000], color="r", label="PEI Peak")
plt.plot([epoxy_peak, epoxy_peak], [0, 5000], color="g", label="EPO Peak")

plt.legend()
plt.show()