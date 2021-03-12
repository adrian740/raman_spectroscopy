import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from os import listdir
from os.path import isfile, join

# Format matplotlib
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams.update({'figure.autolayout': True})

def format_plot():
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')

# Path containing the Raman Analysis
path = "data analysis project//160oC//"

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
#   0. Intensity v Raman Shift
#   1. Raw Epoxy Concentration
#   2. Scaled Concentration Profile
#   3. Demonstration of scan
#   4. Smoothed Concentration Profile
plot_ = [0, 0, 1, 1, 1]

# How many files to skip to plot one
skip = 50
idx = 0

# Dictionaries containing data
mapping = {}
data_dict = {}

# Raman Analysis Settings
#   PEAK: Main peak location
#   WIDTH: width of max search
pei_peak = 1005
pei_width = 10

epoxy_peak = 985
epoxy_width = 10

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
    if plot_[0] is 1 and idx % skip is 0:
        plt.plot(shift, smooth_intensity_data, label=filename)
    data_dict[i] = [shift, intensity, smooth_intensity_data, x_coord]

if plot_[0] is 1:
    plt.xlabel(r"Raman Shift [$cm^{-1}$]")
    plt.ylabel("Intensity [units]")

    format_plot()
    plt.legend()
    plt.show()

# Obtain Raman Max in range
pei_max = []
epo_max = []
x_val = []

for i in ordered_keys:
    shift, intensity, smooth_intensity_data, x_coord = data_dict.get(i)

    # PEI Analysis
    pei_max.append((max(intensity[(shift > pei_peak - pei_width / 2) & (shift < pei_peak + pei_width / 2)])))

    # EPOXY Analysis
    epo_max.append((max(intensity[(shift > epoxy_peak - epoxy_width / 2) & (shift < epoxy_peak + epoxy_width / 2)])))

    # x value
    x_val.append(x_coord)

# Convert to numpy arrays
x_val = np.array(x_val)
pei_max = np.array(pei_max)
epo_max = np.array(epo_max)

# Plot the concentration profiles
if plot_[1] is 1:
    plt.plot(x_val, epo_max, color="C0", label="EPOXY")
    plt.plot(x_val, pei_max, color="C1", label="PEI")

    plt.xlabel("Distance [microns]")
    plt.ylabel("Peak Intensity [units]")

    format_plot()
    plt.legend()
    plt.show()

mask_low_x = (x_val > -50) & (x_val < -37)
mask_high_x = (x_val >= 60) & (x_val < 70)

# PEI
pei_min_mean = np.mean(pei_max[mask_low_x])
pei_max_mean = np.mean(pei_max[mask_high_x])

# EPO
epo_min_mean = np.mean(epo_max[mask_high_x])
epo_max_mean = np.mean(epo_max[mask_low_x])

# Scale between (almost) 0 and 1
ramp_pei = (pei_max - pei_min_mean)/(pei_max_mean - pei_min_mean)
ramp_epo = (epo_max - epo_min_mean)/(epo_max_mean - epo_min_mean)

if plot_[2] is 1:
    plt.plot(x_val, ramp_epo, color="C0", label="EPO")
    plt.plot(x_val, ramp_pei, color="C1", label="PEI")

    plt.xlabel("Distance [microns]")
    plt.ylabel("Normalized Peak Intensity [-]")

    format_plot()
    plt.legend()
    plt.show()

if plot_[3] is 1:
    # Test plot the concentration for x = -40 and 70
    for i in ordered_keys:
        shift, intensity, smooth_intensity_data, x_coord = data_dict.get(i)

        if abs(x_coord + 40.0) < 0.01 or abs(x_coord - 70.0) < 0.01:
            plt.plot(shift, intensity, label=("x = " + str(x_coord)))

    max_v, min_v = 5000, 0

    #plt.plot([pei_peak, pei_peak], [min_v, max_v], color="r", label="PEI Peak: Central")
    plt.plot([pei_peak - pei_width / 2, pei_peak - pei_width / 2], [min_v, max_v], color="r", label="PEI: Lower Bound")
    plt.plot([pei_peak + pei_width / 2, pei_peak + pei_width / 2], [min_v, max_v], color="r", label="PEI: Upper Bound")

    #plt.plot([epoxy_peak, epoxy_peak], [min_v, max_v], color="g", label="EPO Peak: Central")
    plt.plot([epoxy_peak - epoxy_width / 2, epoxy_peak - epoxy_width / 2], [min_v, max_v], color="g", label="EPO: Lower Bound")
    plt.plot([epoxy_peak + epoxy_width / 2, epoxy_peak + epoxy_width / 2], [min_v, max_v], color="g", label="EPO: Upper Bound")

    plt.xlabel(r"Raman Shift [$cm^{-1}$]")
    plt.ylabel("Intensity [units]")

    plt.xlim(950, 1040)

    format_plot()
    plt.legend()
    plt.show()

if plot_[4] is 1:
    # Smoothing
    pei_hat = signal.savgol_filter(ramp_pei, 21, 3)
    epo_hat = signal.savgol_filter(ramp_epo, 21, 3)

    plt.plot(x_val, epo_hat, color="C0", label="EPO")
    plt.plot(x_val, pei_hat, color="C1", label="PEI")

    plt.xlabel("Distance [microns]")
    plt.ylabel("Normalized Peak Intensity [-]")

    format_plot()
    plt.legend()
    plt.show()

# Remaining Questions:
#   Why do they reach similar maximum value (ie same order of magnitude)
#   Why the selected value for cm^-1? - use papers, specifically TUDelft Gradient Tg
#   How to normalize it? - ie values in example are not exactly between 0 and 1
#   Process + make the graph better