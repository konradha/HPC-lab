import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv, exit
try:
    if argv[1] is None:
        print("need to put in valid task number argument")
except Exception as e:
    print("need to put in valid task number argument")
    exit(1)
    #print(e)

data = np.loadtxt("raw_data.txt")
df = pd.DataFrame(data=data, columns=["threads", "nx","ny", "tasks", "t_work", "t_recv", "imb_recv", "imb_work"])
ranks = [2,3,4]
tasks = [10,50,100,200]
ns = [101,1001,2001,4001]
df.sort_values(by=[ "tasks"])
### enter how many tassks here
task = int(argv[1])
eps = .3
dt = df.where((df["tasks"] <= task+eps)).sort_values(by="tasks")
dt = dt.where((df["tasks"] >= task-eps)).sort_values(by="tasks").dropna()
dt = dt.sort_values(by=["nx","threads"])
dt = dt.values.T
for i, val in enumerate(dt[0]):
    if int(val) == 2: 
        plt.plot(dt[0][i:i+3], dt[4][i:i+3],marker='x', linewidth=.7,label=f"nx={int(dt[1][i])}")
plt.grid()
plt.xlabel("ranks / [1]")
plt.ylabel("time / [s]")
plt.xticks([2,3,4])
plt.legend()
plt.title(f"{task} tasks")

plt.savefig(f"task_{task}.png",dpi=400)



"""
eps = .3
for i, task in enumerate(tasks):
    if i is not 10 and i!= 200:
        dt = df.where((df["tasks"] <= tasks[i] + eps)).sort_values(by="tasks")
        dt = dt.where((df["tasks"] >= tasks[i] - eps)).sort_values(by="tasks")
        dt = dt.sort_values(by=["nx","threads"])
        dt = dt.dropna().values.T
        for j, val in enumerate(dt[0]):
            if int(val) == 2:
                plt.plot(dt[0][i:i+3], dt[4][i:i+3])
        plt.xticks([2,3,4])
        plt.show()
        dt = None
"""


"""
dt = df.where((df["tasks"] <=11)).sort_values(by="tasks").dropna().values.T
for i, val in enumerate(dt[0]):
    if int(val) == 2: 
        plt.plot(dt[0][i:i+3], dt[4][i:i+3],)
plt.xticks([2,3,4])
plt.show()
"""
