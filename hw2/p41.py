#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def load_data(x_file, y_file):
    X = np.genfromtxt(x_file, delimiter=',')
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    Y = np.genfromtxt(y_file, delimiter=',')
    return (X, Y)

def sqnorm2(x):
    return np.inner(x, x)

def ridge_regression(X, Y, b, lambd):
    return 1/(2*X.shape[0]) * sqnorm2(X.dot(b) - Y) \
            + lambd / 2 * (sqnorm2(b) - b[0] * b[0])

def ridge_regression_grad(X_batch, Y_batch, b, lambd):
    batch = X_batch.shape[0]
    grad = (X_batch.T).dot(X_batch.dot(b) - Y_batch) + batch * lambd * b
    grad[0] -= batch * lambd * b[0]
    return grad

def sgd(X, Y, b0, lambd, t, batch, epochs, objective, gradient):
    #b = np.random.rand(X.shape[1])
    b = b0.copy()
    fs = np.empty(epochs + 1)
    idx_arr = np.random.randint(0, (batch + X.shape[0] - 1) // batch, epochs)
    fs[0] = objective(X, Y, b, lambd)
    for k, idx in enumerate(idx_arr):
        begin = idx * batch
        end = min(idx * batch + batch, X.shape[0])
        grad = gradient(X[begin:end], Y[begin:end], b, lambd)
        b -= t/batch * grad
        fs[k+1] = objective(X, Y, b, lambd)
    return (b, fs)


if __name__ == "__main__":
    X, Y = load_data('X_train.csv', 'Y_train.csv')
    lambd = 1
    batchs = [10, 20, 50, 100]
    step_sizes = [.01, .001, .0001, .00001]
    epochs = 500
    fstar = 57.0410

    fig = plt.figure(figsize=(4,4))
    plt.subplots_adjust(hspace=0.3)

    b = np.random.rand(X.shape[1])
    for bi, batch in enumerate(batchs):
        for ti, t in enumerate(step_sizes):
            print("batch = {}, step = {}".format(batch, t))
            (b2, fs) = sgd(X, Y, b, lambd, t, batch, epochs,
                          ridge_regression, ridge_regression_grad)
            print("last epoch, beta = {}".format(b))
            print("last epoch, f = {}, min f = {}\n".format(fs[-1], min(fs)))
            plot = fig.add_subplot(len(batchs), len(step_sizes),
                                   1 + bi * len(step_sizes) + ti)
            plot.set_title("b = {}, t = {}".format(batch, t))
            #plot.set_xlim(0, epochs)
            plot.set_yscale('log')
            plot.set_ylim(10**-0, 500)
            plt.plot(fs - fstar)

    plt.show()
