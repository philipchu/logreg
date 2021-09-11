# logistic regression example
# functions taken from (https://beckernick.github.io/logistic-regression-from-scratch/)


import numpy as np
import matplotlib.pyplot as plt


def gen_data():
    np.random.seed()
    num_observations = 1000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations)))
    return simulated_separableish_features, simulated_labels


def plot_data(simulated_separableish_features, simulated_labels):
    plt.figure(figsize=(12, 8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c=simulated_labels, alpha=.4)
    plt.show()


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print
            log_likelihood(features, target, weights)

    return weights


def run_preds(simulated_separableish_features, simulated_labels, add_intercept=True, num_steps=1000):
    weights = logistic_regression(simulated_separableish_features, simulated_labels,
                                  num_steps=num_steps, learning_rate=5e-5, add_intercept=add_intercept)
    if add_intercept:
        simulated_separableish_features = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                                     simulated_separableish_features))
    final_scores = np.dot(simulated_separableish_features, weights)
    preds = np.round(sigmoid(final_scores))
    acc = (preds == simulated_labels).sum().astype(float) / len(preds)
    print('Accuracy from scratch: {0}'.format(acc))
    return preds, acc
