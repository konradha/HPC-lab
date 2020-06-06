import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = np.loadtxt("prelim.txt")
dt = data.T
for i, val in enumerate(dt[0]):
    if dt[1][i] != dt[1][i-1]:
        plt.plot(dt[0][i:i+4], dt[2][i:i+4], label=f"N={int(dt[1][i])}", marker='x',  linewidth=.7)

plt.grid()
plt.xlabel("ranks / [1]")
plt.ylabel("time / [s]")
plt.xticks([1,2,3,4])
plt.legend()
plt.savefig("miniapp_py.png", dpi=400)
