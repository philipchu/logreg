# Description # The aim of this project is to build logistic regression from scratch following  (
# https://web.stanford.edu/~hastie/ElemStatLearn//printings/ESLII_print12_toc.pdf) and building all functions from
# basics. The set of functions included in this file will run error calculation and weight minimization given a set
# of inputs and labels.
# starting with the Y having 2 possible values
# X having LxN dimensionality. [x1 ... xN], where each xi is in itself [xi,1 ... xi,L]'.

import numpy as np


# fitting B_vector to training data (x & y's)
def update_B(B_old, X, Y, learning_rate=.001):
    # Note: B_vector is a [Lx1] structure where B*xi is the linearized basis for the guess.
    # Note: Assuming everything is already an np.matrix
    p_k = predictions(B_old, X)  # p_k is an Nx1 matrix
    W = prob_weighted_diag(p_k)  # NxN diagonal matrix
    hess = hessian(X, W)
    hess_inv = np.linalg.inv(hess)
    error = Y - p_k
    B_new = B_old - learning_rate * np.matmul(np.matmul(hess_inv, X), error)
    return B_new


def hessian(X, W):
    # returns the Hessian of the likelihood function given X & W
    # X is LxN, W is NxN. Hessian is LxL
    return np.matmul(np.matmul(X, W), np.transpose(X))


def predictions(B, X):
    # denom = 1.0
    pred = np.matmul(np.transpose(X), B)
    for row in range(pred.shape[0]):
        pred[row, 0] = sigmoid(pred[row, 0])
    return pred


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def prob_weighted_diag(p_k):
    W = np.matrix(np.zeros(p_k.shape[0]))
    W = np.matmul(np.transpose(W), W)

    for k in range(p_k.shape[0]):
        W[k, k] = p_k[k] * (1 - p_k[k])
    return W


def run_preds(X, Y, add_intercept=True, num_steps=1):
    N = X.shape[1]
    if add_intercept:
        X = np.vstack((np.ones(N), X))
    B = np.transpose(np.matrix(np.zeros(X.shape[0])))
    N_curr = 0
    batch_size = int(np.ceil(N / num_steps))
    while N_curr < N:
        X_batch = X[:, N_curr:(N_curr + batch_size)]
        Y_batch = Y[N_curr:(N_curr + batch_size), :]
        B = update_B(B, X_batch, Y_batch, learning_rate=1/num_steps)
        N_curr = N_curr + batch_size
    preds = np.round(predictions(B, X))
    print('Accuracy from pc: {0}'.format((preds != Y).sum().astype(float) / preds.size))
    return preds


def main():
    B = np.transpose(np.matrix([1.0, 0.0]))
    X = np.matrix([[1.0, -1.0, 2.0], [0.0, 1.0, 1.5]])
    Y = np.transpose(np.matrix([0.0, 1.0, 0.0]))
    P = predictions(B, X)
    B = update_B(B, X, Y, learning_rate=.1)
    print(P)
    print(B)
