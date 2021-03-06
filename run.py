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

    range = data[:, 0]
    shift = data[:, 1]

    # Butterworth Filter
    N = 3  # Filter order
    Wn = 0.1  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B, A, shift)

    # raman shift, amplitude, + smoothed amplitude
    return range, shift, smooth_data

#
# SETTINGS
#

# Plot settings
plot_ = True

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
    range_, raw_data, smooth_data = read_data(path + filename)
    idx += 1
    if plot_ and idx % skip is 0:
        plt.plot(range_, smooth_data, label=filename)
    data_dict[i] = [range_, raw_data, smooth_data]

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

for i in ordered_keys:
    range_, raw_data, smooth_data = data_dict.get(i)

    # PEI Analysis
    pei_idx_low, piw_val_low = find_nearest(range_, pei_peak - pei_width / 2)
    pei_idx_high, piw_val_high = find_nearest(range_, pei_peak + pei_width / 2)

    pei_dat = raw_data[pei_idx_high:pei_idx_low]
    pei_max.append(max(pei_dat))

    # EPOXY Analysis
    epo_idx_low, piw_epo_low = find_nearest(range_, epoxy_peak - epoxy_width / 2)
    epo_idx_high, piw_epo_high = find_nearest(range_, epoxy_peak + epoxy_width / 2)

    epo_dat = raw_data[epo_idx_high:epo_idx_low]
    epo_max.append(max(epo_dat))

# Plot the concentration profiles
plt.plot(pei_max, label="PEI")
plt.plot(epo_max, label="EPOXY")
plt.legend()
plt.show()

# Remaining Questions:
#   Why do they reach similar maximum
#   Why the selected value for cm^-1?
#   How to normalize it?
#   Process + make the graph better
#   Rename shift + range