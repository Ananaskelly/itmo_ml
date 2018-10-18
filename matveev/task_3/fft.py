from matveev.task_3.dataset import Dataset


import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


ds = Dataset()
x_train, y_train, x_test, y_test = ds.split_train_test_set()
num_train, h, w = x_train.shape
num_test, _, _ = x_test.shape


x_train_fft = []
for i in range(num_train):
    x_train_fft.append(np.abs(np.fft.rfft(np.reshape(x_train[i, :, :], newshape=(h*w)))))

x_train_fft = np.array(x_train_fft)

x_test_fft = []
for i in range(num_test):
    x_test_fft.append(np.abs(np.fft.rfft(np.reshape(x_test[i, :, :], newshape=(h*w)))))

x_test_fft = np.array(x_test_fft)


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train_fft, y_train)

pred_labels = neigh.predict(x_test_fft)

print(accuracy_score(pred_labels, y_test))
