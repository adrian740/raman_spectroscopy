import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("Benitoite__R050320__Broad_Scan__780__0__unoriented__Raman_Data_RAW__17845.txt", comments="##", delimiter=",")

range = data[:,0]
shift = data[:,1]

max_shift = max(abs(shift))
shift = shift / max_shift

plt.plot(range, shift)
plt.show()