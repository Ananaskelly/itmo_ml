import csv
import numpy as np


cat_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}


class Dataset:

    def __init__(self, path_to_train, path_to_test):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test

        self.num_feats = 20
        self.cat_feats = 7

    def load_ds(self, ds_type='train'):

        if ds_type == 'train':
            path = self.path_to_train
        elif ds_type == 'test':
            path = self.path_to_test
        else:
            raise Exception('Can\'t load data! Such ds type is not supported!')

        with open(path, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')

            # skip header
            next(reader)

            x = []
            y = []

            for row in reader:
                feats_row = row[1:1 + self.num_feats]

                curr_idx = 1 + self.num_feats
                for i in range(self.cat_feats):
                    cat_label = row[curr_idx]
                    feats_row.append(cat_dict[cat_label])
                    curr_idx += 1

                x.append(feats_row)

                cat_label = row[-1]
                y.append(cat_dict[cat_label])

            x = np.asarray(x, dtype=float)
            y = np.asarray(self.dense_to_one_hot(y, 7), dtype=float)

        return x, y

    def dense_to_one_hot(self, labels_dense, num_classes):
        labels_one_hot = np.zeros(shape=(len(labels_dense), num_classes), )
        for i in range(len(labels_dense)):
            labels_one_hot.itemset((i, labels_dense[i]), 1)
        return labels_one_hot
