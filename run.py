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

# Path containing the Raman Spectrum
path = "data analysis project//180oC//"

# All of the files in the directory (only analyze .txt files)
allfiles = [f for f in listdir(path) if isfile(join(path, f))]

# Filter
def butterworth_filter(dat):
    # Butterworth Filter
    N = 3  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    return signal.filtfilt(B, A, dat)

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
plot_ = [1, 1, 1, 1, 1, 1]

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
pei_width = 10

epoxy_peak = 984
epoxy_width = 14

plot_red_lines = True

# Places where it is known that the concentration is 100% for one material and 0% for the other:
# Test plot x coordinate, peak, width, color

# Used for single layer analysis
location1 = [-50, -37]
location2 = [60, 70]
test_plot = np.array([[-40.0, epoxy_peak, epoxy_width, "C0"], [70.0, pei_peak, pei_width, "C1"]])

# Used for multi layer analysis
#location1 = [53500, 53560]
#location2 = [52920, 52950]
#test_plot = np.array([[52791.6, epoxy_peak, epoxy_width, "C0"], [53715.6, pei_peak, pei_width, "C1"]])

#
# End Raman Analysis Settings
#

# Read in data
for filename in allfiles:
    if filename.endswith(".txt"):
        mapping[filename.split("_")[-7]] = filename
ordered_keys = sorted(mapping, key = lambda x: float(x))

# Actually plot (+ process data)
for i in ordered_keys:
    filename = mapping[i]
    shift, intensity, smooth_intensity_data, x_coord = read_data(path + filename)
    idx += 1
    if plot_[0] == 1 and idx % skip == 0:
        plt.plot(shift, smooth_intensity_data, label=filename)
    data_dict[i] = [shift, intensity, smooth_intensity_data, x_coord]

if plot_[0] == 1:
    plt.xlabel(r"Raman Shift [$cm^{-1}$]")
    plt.ylabel("Intensity [Counts]")

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

# Two Extremes
mask_low_x = (x_val > min(location1)) & (x_val < max(location1))
mask_high_x = (x_val >= min(location2)) & (x_val < max(location2))

# PEI
pei_min_mean = min(np.mean(pei_max[mask_low_x]), np.mean(pei_max[mask_high_x]))
pei_max_mean = max(np.mean(pei_max[mask_low_x]), np.mean(pei_max[mask_high_x]))

# EPO
epo_min_mean = min(np.mean(epo_max[mask_high_x]), np.mean(epo_max[mask_low_x]))
epo_max_mean = max(np.mean(epo_max[mask_high_x]), np.mean(epo_max[mask_low_x]))

# Scale between (almost) 0 and 1
ramp_pei = (pei_max - pei_min_mean)/(pei_max_mean - pei_min_mean)
ramp_epo = (epo_max - epo_min_mean)/(epo_max_mean - epo_min_mean)

# Plot the concentration profiles
if plot_[1] == 1:
    plt.plot(x_val, epo_max, color="C0", label="% Epoxy")
    plt.plot(x_val, pei_max, color="C1", label="% PEI")
    
    if plot_red_lines: # Won't work for some data
        plt.plot(location1, [epo_max_mean, epo_max_mean], color="r")
        plt.plot(location1, [pei_min_mean, pei_min_mean], color="r")
        
        plt.plot(location2, [epo_min_mean, epo_min_mean], color="r")
        plt.plot(location2, [pei_max_mean, pei_max_mean], color="r")

    plt.xlabel("Distance [micrometers]")
    plt.ylabel("Peak Intensity [Counts]")

    format_plot()
    plt.legend()
    plt.show()

if plot_[2] == 1:
    plt.plot(x_val, ramp_epo, color="C0", label="% Epoxy")
    plt.plot(x_val, ramp_pei, color="C1", label="% PEI")

    plt.xlabel("Distance [micrometers]")
    plt.ylabel("Normalized Peak Intensity [-]")
    
    plt.ylim(-0.1, 1.2)

    format_plot()
    plt.legend()
    plt.show()

def root(x1, y1, x2, y2):
    return x1 - y1 * (x2 - x1) / (y2 - y1)

def get_FWHM(shift, peak, width, intensity, x):
    mask = (shift > peak - width / 2) & (shift < peak + width / 2)
    max_s = max(intensity[mask]) / 2
    diff = intensity - max_s
    root_area = shift[mask & (diff > 0)]
    roots = root_area[-1], root_area[0]
    idx_roots = np.where(shift == roots[0])[0][0], np.where(shift == roots[1])[0][0]

    root1 = root(shift[idx_roots[0] + 1], diff[idx_roots[0] + 1], shift[idx_roots[0]], diff[idx_roots[0]])
    root2 = root(shift[idx_roots[1]], diff[idx_roots[1]], shift[idx_roots[1] - 1], diff[idx_roots[1] - 1])

    return min(root1, root2), max(root1, root2), max_s

if plot_[3] == 1:
    for i in ordered_keys:
        shift, intensity, smooth_intensity_data, x_coord = data_dict.get(i)

        difference = test_plot[:, 0].astype(np.float) - x_coord
        idxs = np.where(abs(difference) < 0.01)[0]

        if len(idxs) != 0:
            focus_peak, focus_width, color = test_plot[:, 1][idxs], test_plot[:, 2][idxs], test_plot[:, 3][idxs][0]
            plt.plot(shift, intensity, label=("x = " + str(x_coord)), color=color)
            low_s, high_s, max_s = get_FWHM(shift, float(focus_peak), float(focus_width), intensity, x_coord)
            plt.plot([low_s, low_s, high_s, high_s], [0, max_s, max_s, 0], color="fuchsia")
            print("x:", x_coord, "FWHM:", round(high_s - low_s, 2), "Min/Max bounds:", round(low_s, 2), round(high_s, 2))

    max_v, min_v = 5000, 0

    plt.plot([pei_peak - pei_width / 2, pei_peak - pei_width / 2], [min_v, max_v], color="r", label="PEI Bounds")
    plt.plot([pei_peak + pei_width / 2, pei_peak + pei_width / 2], [min_v, max_v], color="r")

    plt.plot([epoxy_peak - epoxy_width / 2, epoxy_peak - epoxy_width / 2], [min_v, max_v], color="g", label="Epoxy Bounds")
    plt.plot([epoxy_peak + epoxy_width / 2, epoxy_peak + epoxy_width / 2], [min_v, max_v], color="g")

    plt.xlabel(r"Raman Shift [$cm^{-1}$]")
    plt.ylabel("Intensity [Counts]")

    plt.xlim(950, 1040)
    plt.ylim(min_v, max_v)

    format_plot()
    plt.legend()
    plt.show()

if plot_[4] == 1:
    # Smoothing
    pei_hat = butterworth_filter(ramp_pei)
    epo_hat = butterworth_filter(ramp_epo)

    plt.plot(x_val, epo_hat, color="C0", label="% Epoxy")
    plt.plot(x_val, pei_hat, color="C1", label="% PEI")

    plt.xlabel("Distance [micrometers]")
    plt.ylabel("Normalized Peak Intensity [-]")

    format_plot()
    plt.legend()
    plt.show()
    
if plot_[5] == 1:
    # Smoothing Sum
    pei_hat = butterworth_filter(ramp_pei)
    epo_hat = butterworth_filter(ramp_epo)

    plt.plot(x_val, epo_hat + pei_hat - 1, color="C0", label="Deviation from 100%")

    plt.xlabel("Distance [micrometers]")
    plt.ylabel("Deviation [-]")

    format_plot()
    plt.legend()
    plt.show()