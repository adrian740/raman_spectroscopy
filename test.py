import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

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

def plot_data(filename):
    shift, intensity, smooth, x_coord = read_data(filename)
    plt.plot(shift, intensity, label="x = " + str(x_coord))

max_v, min_v = 5000, 0
plt.plot([1005, 1005], [min_v, max_v], color="r", label="PEI Peak")
plt.plot([985, 985], [min_v, max_v], color="g", label="EPO Peak")

plot_data("data analysis project//120oC//120 80_0__X_-50__Y_-2.35417__Time_0.txt")
plot_data("data analysis project//120oC//120 80_298__X_69.2__Y_-2.35417__Time_1154.txt")

plt.ylabel("Intensity [units]")
plt.xlabel(r"Raman Shift [$cm^{-1}$]")

plt.legend()
plt.show()