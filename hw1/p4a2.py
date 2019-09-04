#!/usr/bin/env python3

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math

def lasso_solve(input_file, lambdas, pnorm=2):
    toy = np.loadtxt(open(input_file, "rb"), delimiter=",")
    m, n = toy.shape
    nval = len(lambdas)

    fig = plt.figure(figsize=(8,8))
    figr = math.ceil((1 + nval * 2) / 5)
    figc = min(5, 1 + nval*2)
    plt.subplots_adjust(hspace=0.3)

    plot = fig.add_subplot(figr, figc, 1)
    plot.set_title("Original")
    plt.imshow(toy)

    theta = cp.Variable(toy.shape)
    lambd = cp.Parameter(nonneg=True)

    hdiff = cp.diff(theta, axis=0)
    vdiff = cp.diff(theta, axis=1)

    hdiff_flat = cp.reshape(hdiff[0:n-1, 0:m-1], ((n-1)*(m-1),))
    vdiff_flat = cp.reshape(vdiff[0:n-1, 0:m-1], ((n-1)*(m-1),))
    stack = cp.vstack([hdiff_flat, vdiff_flat])

    objective = cp.sum_squares(toy - theta)/2 + lambd * cp.sum(
        cp.norm(stack, p=pnorm, axis=0))
    problem = cp.Problem(cp.Minimize(objective))

    for idx, val in enumerate(lambdas):
        lambd.value = val
        problem.solve(warm_start=True)
        print("Lambda: {}, objective value: {:.2f}".format(val, problem.value))
        plot = fig.add_subplot(figr, figc, 2 + 2*idx)
        plot.set_title("Lambda {}".format(val))
        plt.imshow(theta.value)

        # histogram
        plot = fig.add_subplot(figr, figc, 3 + 2*idx)
        plt.hist(theta.value, bins=100, histtype='step')
    plt.show()


if __name__ == "__main__":
    lasso_solve("toy.csv", [1])
