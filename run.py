import matplotlib.pyplot as plt
import numpy as np

file = "data analysis project//120oC/120 80_1__X_-49.6__Y_-2.35417__Time_3.txt"

data = np.genfromtxt(file, delimiter="\t")

range = data[:,0]
shift = data[:,1]

plt.plot(range, shift, label="Raw Data")

plt.legend()
plt.show()