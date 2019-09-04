import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def neighbor_abs(array, m, n):
    for i in range(m):
        for j in range(n):
            if j < n - 1 and i < m - 1:
                yield cp.abs(array[i*n+j] - array[i*n+j+1])
                yield cp.abs(array[i*n+j] - array[i*n+n+j])
            #if j < n - 1:
            #    yield cp.abs(array[i*n+j] - array[i*n+j+1])
            #if i < m - 1:
            #    yield cp.abs(array[i*n+j] - array[i*n+n+j])

toy = np.loadtxt(open("toy.csv", "rb"), delimiter=",")
m, n = toy.shape
fig = plt.figure(figsize=(8,4))
fig.add_subplot(1,2,1)
plt.imshow(toy)

theta = cp.Variable(toy.size)
lambd = cp.Parameter(nonneg=True)
objective = cp.sum_squares(toy.flatten() - theta)/2 + lambd * sum(neighbor_abs(theta, m, n))
problem = cp.Problem(cp.Minimize(objective))
lambd.value = 1
problem.solve()

print("Objective value: {:.2f}".format(problem.value))

fig.add_subplot(1,2,2)
plt.imshow(theta.value.reshape(toy.shape))

plt.show()
