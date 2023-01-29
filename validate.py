import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

import sys
time = sys.argv[1][8:]
npzfile = np.load("archive/"+time+"/speedup_"+time+".npz")
W, H, speedup = npzfile["W"], npzfile["H"], npzfile["speedup"]

X = np.empty(shape=(W.flatten().shape[0], 2))
X[:, 0] = W.flatten()
X[:, 1] = H.flatten()
speedup = speedup.flatten()

model = LinearRegression()
model.fit(X, speedup)

print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)

speedup_predicted =  model.coef_[0]*X[:,0] + model.coef_[1]*X[:,1] + model.intercept_


print("Rel Err Std: ", np.std(np.abs(speedup_predicted - speedup)/speedup))
fig, ax = plt.subplots()
ax.scatter(range(len(speedup)), np.abs(speedup_predicted - speedup)/speedup)
ax.set_xticks(range(len(speedup)))
ax.set_xticklabels([str(item) for item in itertools.product(W[0], H[0])])
plt.xticks(rotation=90)
fig.savefig("archive/"+time+"/rel_err_vs_dims_"+time+".png")



