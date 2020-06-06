import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.grid()

data = "data.txt"
data = np.loadtxt(data)
dt = data.T
m = {'128': [], '256': [], '512': [], '1024': []}

df = pd.DataFrame(data=data, columns=['N', 'ranks', 'time'])
df = df.sort_values(by='N')
dt = df.values.T
cs = ["#000000","#00FF00","#0000FF","#FF0000"]
count = 0
for i, val in enumerate(dt[1]):   
    if int(val) == 1:
        plt.scatter(dt[1][i:(i+4)], dt[2][i:(i+4)], label=f'N={int(dt[0][i])}', color=cs[count])
        plt.plot(dt[1][i:(i+4)], dt[2][i:(i+4)], color=cs[count])
        count += 1

plt.xlabel("ranks / [1]")
plt.ylabel("time / [s]")
#plt.xticks([128, 256, 512, 1024])
plt.xticks([1,2,3,4])
plt.yscale("log")
plt.legend()
plt.savefig("miniapp_c.png", dpi=400)
