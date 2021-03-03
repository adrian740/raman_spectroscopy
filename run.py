import matplotlib.pyplot as plt
import numpy as np
from BaselineRemoval import BaselineRemoval

data = np.genfromtxt("Benitoite__R050320__Broad_Scan__532__0__unoriented__Raman_Data_RAW__17844.txt", comments="##", delimiter=",")

range = data[:,0]
shift = data[:,1]

max_shift = max(abs(shift))
shift = shift / max_shift

# See https://pypi.org/project/BaselineRemoval/
dat = BaselineRemoval(shift)
Zhangfit_output=dat.ZhangFit()

plt.plot(range, shift)
plt.plot(range, Zhangfit_output)
plt.show()