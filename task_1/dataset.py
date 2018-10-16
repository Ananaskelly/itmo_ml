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

cat_dict_INV = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G'
}


class Dataset:

    def __init__(self, path_to_train, path_to_test):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test

        self.num_feats = 20
        self.cat_feats = 7

        self.valid_part = 0.1

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
            x = self.normalize(x)
            y_class = np.asarray(y)
            y = np.asarray(self.dense_to_one_hot(y, 7), dtype=float)

        return x, y_class, y

    def normalize(self, data):
        min_val = np.min(data[:, :self.num_feats], axis=0)
        max_val = np.max(data[:, :self.num_feats], axis=0)

        data[:, :self.num_feats] = (data[:, :self.num_feats] - min_val)/(max_val - min_val)

        return data

    def load_test(self):

        with open(self.path_to_test, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')

            # skip header
            next(reader)

            x = []

            for row in reader:
                feats_row = row[1:1 + self.num_feats]

                curr_idx = 1 + self.num_feats
                for i in range(self.cat_feats):
                    cat_label = row[curr_idx]
                    feats_row.append(cat_dict[cat_label])
                    curr_idx += 1

                x.append(feats_row)

            x = np.asarray(x, dtype=float)
            x = self.normalize(x)

        return x

    def load_ds_with_valid(self):
        x, _, y = self.load_ds(ds_type='train')

        num_ex, num_feats = x.shape

        perm = np.arange(num_ex)
        np.random.shuffle(perm)

        x = x[perm]
        y = y[perm]

        bound = int(num_ex*(1 - self.valid_part))

        return x[:bound, :], y[:bound], x[bound:], y[bound:]

    def load_ds_with_valid_not_one_hot(self):
        x, y, _ = self.load_ds(ds_type='train')
        num_ex, num_feats = x.shape

        perm = np.arange(num_ex)
        np.random.shuffle(perm)

        x = x[perm]
        y = y[perm]

        bound = int(num_ex * (1 - self.valid_part))

        return x[:bound, :], y[:bound], x[bound:], y[bound:]

    def dense_to_one_hot(self, labels_dense, num_classes):
        labels_one_hot = np.zeros(shape=(len(labels_dense), num_classes), )
        for i in range(len(labels_dense)):
            labels_one_hot.itemset((i, labels_dense[i]), 1)
        return labels_one_hot

    def one_hot_to_dense(self, labels_dense):

        return np.argmax(labels_dense, axis=1)

    def test_to_csv(self, filename, results):

        with open(filename, 'w'):
            pass

        with open(filename, 'a+') as csv_log_res:
            csv_log_res.write('id,class\n')

            idx = 1
            for res in results:
                csv_log_res.write(str(idx) + ',' + cat_dict_INV[res] + '\n')
                idx += 2
