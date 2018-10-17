from matveev.task_3.dataset import Dataset

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog


ds = Dataset()
x_train, y_train, x_test, y_test = ds.split_train_test_set()
num_train, h, w = x_train.shape
num_test, _, _ = x_test.shape

x_train_hogs = []
for i in range(num_train):
    hog_image = hog(x_train[i, :, :], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    x_train_hogs.append(hog_image)

x_train_hogs = np.array(x_train_hogs)

x_test_hogs = []
for i in range(num_test):
    hog_image = hog(x_test[i, :, :], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    x_test_hogs.append(hog_image)

x_test_hogs = np.array(x_test_hogs)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train_hogs, y_train)

pred_labels = neigh.predict(x_test_hogs)

print(accuracy_score(pred_labels, y_test))
