import os
import numpy as np
from sklearn.decomposition import PCA


ROOT_PATH = '../data/'
TR_DATA = 'arcene_train.data'
TR_LABELS = 'arcene_train.labels'
VALID_DATA = 'arcene_valid.data'
VALID_LABELS = 'arcene_valid.labels'


class Dataset:

    def __init__(self, tst_part=0.2):
        self.train_set = None
        self.valid_set = None

        self.nn_train_set = None
        self.nn_valid_set = None

        self.train_ex = 0
        self.feats_num = 0

        self.num_classes = 2
        self.test_part = tst_part

        self.all_ex = None
        self.all_labels = None

    def create_ds(self, pca=False):

        train_data = []
        with open(os.path.join(ROOT_PATH, TR_DATA), 'r') as train_data_file:
            lines = train_data_file.readlines()

            for line in lines:
                train_data.append(self.line2array(line))

            train_data = np.array(train_data)

        train_labels = []
        with open(os.path.join(ROOT_PATH, TR_LABELS), 'r') as train_labels_file:
            lines = train_labels_file.readlines()

            for line in lines:
                train_labels.append(int(line.strip('\n')))

            train_labels = np.array(train_labels)

        assert train_data.shape[0] == train_labels.shape[0], 'Number of train examples and labels must be equal!'

        if pca:
            self.train_set = {
                'data': self.pca(self.normalize_along_axis(train_data)),
                'labels': train_labels
            }
        else:
            self.train_set = {
                'data': self.normalize_along_axis(train_data),
                'labels': train_labels
            }

        self.nn_train_set = {
            'data': train_data,
            'labels': train_labels
        }

        self.train_ex, self.feats_num = self.train_set['data'].shape

        valid_data = []
        with open(os.path.join(ROOT_PATH, VALID_DATA), 'r') as valid_data_file:
            lines = valid_data_file.readlines()

            for line in lines:
                valid_data.append(self.line2array(line))

            valid_data = np.array(valid_data)

        valid_labels = []
        with open(os.path.join(ROOT_PATH, VALID_LABELS), 'r') as valid_labels_file:
            lines = valid_labels_file.readlines()

            for line in lines:
                valid_labels.append(int(line.strip('\n')))

            valid_labels = np.array(valid_labels)

        assert valid_data.shape[0] == valid_labels.shape[0], 'Number of train examples and labels must be equal!'

        if pca:
            self.valid_set = {
                'data': self.pca(self.normalize_along_axis(valid_data)),
                'labels': valid_labels
            }
        else:
            self.valid_set = {
                'data': self.normalize_along_axis(valid_data),
                'labels': valid_labels
            }
        self.nn_valid_set = {
            'data': valid_data,
            'labels': valid_labels
        }

        self.nn_valid_set = {
            'data': valid_data,
            'labels': valid_labels
        }

    @staticmethod
    def normalize_along_axis(data):

        min_ = np.min(data, axis=0)
        max_ = np.max(data, axis=0)

        diff = np.where(max_ - min_ != 0, max_ - min_, max_)
        diff = np.where(diff == 0, 1, diff)

        return (data - min_)/diff

    @staticmethod
    def pca(data):
        pca = PCA()
        data = pca.fit_transform(data)
        return data

    def line2array(self, line, delimiter=' '):
        line = line.strip('\n ')
        symbs = line.split(delimiter)

        arr = np.zeros(shape=len(symbs))
        for i, s in enumerate(symbs):
            arr[i] = int(s)

        return arr

    def shuffle_train_set(self):

        num_ex = self.train_set['data'].shape[0]

        perm = np.arange(num_ex)
        np.random.shuffle(perm)
        self.train_set['data'] = self.train_set['data'][perm]
        self.train_set['labels'] = self.train_set['labels'][perm]

    def get_subset_set(self, idxs, data_type='train'):

        if data_type == 'train':
            return self.train_set['data'][:, idxs], self.train_set['labels']
        elif data_type == 'valid':
            return self.valid_set['data'][:, idxs], self.valid_set['labels']




