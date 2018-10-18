import cv2
import os
import numpy as np


ROOT_PATH = 'C:/Users/avdeeva-a/att_faces/orl_faces'


class Dataset:

    def __init__(self, tst_part=0.2):
        self.train_set = None
        self.test_set = None

        self.num_classes = 0
        self.test_part = tst_part

        self.all_ex = None
        self.all_labels = None

    def create_ds(self):

        dirs = os.listdir(ROOT_PATH)

        self.num_classes = len(dirs)

        all_ex = []
        all_labels = []

        for i, d in enumerate(dirs):
            dir_path = os.path.join(ROOT_PATH, d)
            files = os.listdir(dir_path)
            for f in files:
                all_ex.append(cv2.imread(os.path.join(dir_path, f), 0))
                all_labels.append(i)

        all_ex = np.stack(all_ex)
        all_labels = np.stack(all_labels)

        # all_ex = self.normalize(all_ex)
        # all_labels = self.dense_to_one_hot(all_labels, self.num_classes)

        return all_ex, all_labels

    def split_train_test_set(self, tst_part):
        if self.all_ex is None and self.all_labels is None:
            self.all_ex, self.all_labels = self.create_ds()

        num_ex = self.all_ex.shape[0]

        perm = np.arange(num_ex)
        np.random.shuffle(perm)
        x = self.all_ex[perm]
        y = self.all_labels[perm]

        bound_idx = int(num_ex * (1 - tst_part))

        return x[:bound_idx, :], y[:bound_idx], x[bound_idx:, :], y[bound_idx:]

    def normalize(self, data):
        return data * 1/256

    def dense_to_one_hot(self, labels_dense, num_classes):
        labels_one_hot = np.zeros(shape=(len(labels_dense), num_classes), )
        for i in range(len(labels_dense)):
            labels_one_hot.itemset((i, labels_dense[i]), 1)
        return labels_one_hot
