import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
time = sys.argv[1]
npzfile = np.load("archive/"+time+"/speedup_"+time+".npz")
widths, heights, sizes, speedup = npzfile["widths"], npzfile["heights"], npzfile["sizes"], npzfile["speedup"]

fig, ax = plt.subplots(len(sizes))
sns.heatmap(speedup[0].T, cmap="coolwarm")

ax.set_xticklabels(widths)
ax.set_xlabel("widths")
ax.set_yticklabels(heights)
ax.set_ylabel("heights")

fig.savefig("archive/"+time+"/heatmap_"+time+".png")
