from matveev.task_3.dataset import Dataset


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


ds = Dataset()
x_train, y_train, x_test, y_test = ds.split_train_test_set()
num_test, h, w = x_train.shape
num_train, _, _ = x_test.shape


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(np.reshape(x_train, newshape=(num_test, h*w)), y_train)

pred_labels = neigh.predict(np.reshape(x_test, newshape=(num_train, h*w)))

print(accuracy_score(pred_labels, y_test))
