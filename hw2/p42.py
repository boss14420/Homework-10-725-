#!/usr/bin/env python3
import math
import p41

import numpy as np
import matplotlib.pyplot as plt

def lasso(XbY, b, lambd, ws, gs):
    return p41.sqnorm2(XbY)*XbY.shape[0]/2 + \
        lambd * sum([w * np.linalg.norm(b[g[0]:g[1]])
                         for w, g in zip(ws,gs)])

def lasso_prox(b, t, ws, gs, lambd):
    return np.array([b[0]] +
                    [bj * max(0, 1 - t*w*lambd/np.linalg.norm(bj))
                     for w, g in zip(ws, gs)
                     for bj in b[g[0]:g[1]]])

def lasso_grad(X, XbY):
    return (X.T).dot(XbY)

def lasso_solve(X, Y, ws, gs, bstart, lambd = 0.02, t = 0.005, epochs = 10000):
    b = bstart.copy()
    fs = np.empty(epochs + 1)
    XbY = (X.dot(b) - Y)/X.shape[0]
    fs[0] = lasso(XbY, b, lambd, ws, gs)
    for k in range(1, epochs + 1):
        grad = lasso_grad(X, XbY)
        b = lasso_prox(b - t*grad, t, ws, gs, lambd)
        XbY = (X.dot(b) - Y)/X.shape[0]
        fs[k] = lasso(XbY, b, lambd, ws, gs)
    return b, fs

def lasso_solve_accel(X, Y, ws, gs, bstart, lambd = 0.02, t = 0.005, epochs = 10000):
    b = bstart.copy()
    b0 = b.copy()
    fs = np.empty(epochs + 1)
    XbY = (X.dot(b) - Y)/X.shape[0]
    fs[0] = lasso(XbY, b, lambd, ws, gs)
    for k in range(1, epochs + 1):
        grad = lasso_grad(X, XbY)
        v = b + (k - 2)/(k + 1) * (b - b0)
        b0 = b
        b = lasso_prox(v - t*grad, t, ws, gs, lambd)
        XbY = (X.dot(b) - Y)/X.shape[0]
        fs[k] = lasso(XbY, b, lambd, ws, gs)
    return b, fs

if __name__=="__main__":
    X = np.genfromtxt('X_train.csv', delimiter=',')
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    Y = np.genfromtxt('Y_train.csv', delimiter=',')
    gs = [[1, 2], [2, 3], [3, 8], [8, 14], [14, 16], [16, 17], [17, 18], [18, 19]]
    ws = [math.sqrt(g[1] - g[0]) for g in gs]
    fstar = 49.9649

    lambd = 0.02
    t = 0.005
    epochs = 10000

    bstart = np.random.rand(X.shape[1])

    fig = plt.figure(figsize=(1,2))
    plt.subplots_adjust(hspace=0.3)

    plt.yscale("log")
    plt.ylim(10**-2, 500)

    for idx, func in enumerate([lasso_solve, lasso_solve_accel]):
        name = func.__name__
        print(name)
        b, fs = func(X, Y, ws, gs, bstart, t=t, epochs=epochs)

        print("Min value = {}".format(min(fs)))
        print("beta = {}".format(b))

        plt.plot(fs - fstar, label=name)

    plt.legend(loc='upper right')
    plt.show()
