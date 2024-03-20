import numpy as np
import pandas as pd
from sklearn.svm import SVC


def load_data():
    URL_ = 'mystery.data'
    data = pd.read_csv(URL_, header=None)
    print(data)
    return data


data = load_data()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = y.apply(lambda x: 1 if x == 1 else -1)

model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

print(model.support_vectors_)
print(model.get_params())
print(model.coef_)
print(model.test)