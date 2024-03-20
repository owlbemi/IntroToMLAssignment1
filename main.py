import numpy as np
import pandas as pd


def load_data():
    URL_ = 'perceptron.data'
    data = pd.read_csv(URL_, header=None)
    data = np.asmatrix(data, dtype='float64')
    print(data)
    return data


data = load_data()


# stochastic
def stoc_perceptron(data):
    features = data[:, :-1]
    labels = data[:, -1]

    w = np.ones(shape=(1, features.shape[1] + 1))

    misclassified_ = []
    ploss_ = []
    steps = 0

    while True:
        misclassified = 0
        ploss = 0

        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            y = np.dot(w, x.transpose())
            target = 1.0 if (y > 0) else -1.0

            delta = (label.item(0, 0) - target)
            ploss += max(0, -label.item(0, 0) * y)

            if delta != 0:
                misclassified += 1
                w += delta * x
        misclassified_.append(misclassified)
        ploss_.append(ploss)
        steps += 1

        print(f"Step {steps}: Misclassified = {misclassified}, Perceptron Loss = {ploss}, Weight = {w}")

        if ploss < 10e-8:
            break
    return w, misclassified_, ploss_, steps


w, misclassified_, ploss_, steps = stoc_perceptron(data)

print(ploss_)


# standard

def standard_perceptron(data):
    features = data[:, :-1]
    labels = data[:, -1]

    w = np.ones(shape=(1, features.shape[1] + 1))

    misclassified_ = []
    ploss_ = []
    steps = 0

    while True:
        misclassified = 0
        ploss = 0
        total_delta = np.zeros_like(w)

        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            y = np.dot(w, x.transpose())
            target = 1.0 if y > 0 else -1.0

            delta = (label - target)
            ploss += max(0, -label * y)

            if delta != 0:
                misclassified += 1
                total_delta += delta * x

        misclassified_.append(misclassified)
        ploss_.append(ploss)
        steps += 1

        w += total_delta
        print(f"Step {steps}: Misclassified = {misclassified}, Perceptron Loss = {ploss}, Weight = {w}")

        if ploss < 10e-8:
            break

    return w, misclassified_, ploss_, steps

w, misclassified_, ploss_, steps = standard_perceptron(data)
print(ploss_)

