import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
time = sys.argv[1]
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
npzfile = np.load("archive/"+time+"/speedup_"+time+".npz")
widths, heights, sizes, speedup, graph_weights = npzfile["widths"], npzfile["heights"], npzfile["sizes"], npzfile["speedup"], npzfile["graph_weights"]


x1 = speedup[0,0,0]
x1_p = speedup[0,0,1]

x2 = (speedup[0,1,0] - graph_weights[1][1]*x1)/(graph_weights[1][0])
x2_p = (speedup[0,1,1] - graph_weights[1][1]*x1)/(graph_weights[1][0])

speedup_predicted = np.empty(shape=len(heights)-2)
speedup_actual = np.empty(shape=len(heights)-2)

for i in range(2,len(heights)):
    speedup_predicted[i-2] = (graph_weights[i][-1]*x1 + sum(graph_weights[i][0:-1])*x2)/(graph_weights[i][-1]*x1_p + sum(graph_weights[i][0:-1])*x2_p)
    speedup_actual[i-2] = speedup[0,i,0]/speedup[0,i,1]

fig, ax = plt.subplots(1,2)
ax[0].plot(heights[2:], speedup_actual)
ax[0].set_xticklabels(heights[2:])
ax[0].set_xlabel("heights")
ax[0].set_ylabel("speedup_actual")
ax[0].set_title("Size 1000000")

ax[1].plot(heights[2:], speedup_predicted/speedup_actual)
ax[1].set_xticklabels(heights[2:])
ax[1].set_xlabel("heights")
ax[1].set_ylabel("speedup_predicted/speedup_actual")
ax[1].set_title("Size 1000000")

fig.tight_layout()
fig.savefig("archive/"+time+"/speedup_"+time+".png")
