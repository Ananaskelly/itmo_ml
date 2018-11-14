import numpy as np
from xml.etree import ElementTree
from sint.laba_one.dataset import Dataset as LBDataset


class Dataset(LBDataset):

    def __init__(self, path_to_in_xml, valid_part=0.1, test_part=0.1):
        super(Dataset, self).__init__(path_to_in_xml, valid_part=valid_part, test_part=test_part)

    def create_ds(self, use_th_gr=False):
        words, attrs, intonation = self.parse_xml(return_intonation=True)
        classes = intonation
        feats = self.get_features(words, attrs)

        if use_th_gr:
            feats = self.gen_three_grams(feats)

        feats, classes = Dataset.random_shuffle(feats, classes)
        num_ex = classes.shape[0]

        train_bound = int(num_ex * (1 - self._valid_part - self._test_part))
        valid_bound = int(num_ex * (1 - self._valid_part))

        self._train_set = {
            'data': feats[:train_bound, :],
            'labels': classes[:train_bound]
        }

        self._valid_set = {
            'data': feats[train_bound:valid_bound, :],
            'labels': classes[train_bound:valid_bound]
        }

        self._test_set = {
            'data': feats[valid_bound:, :],
            'labels': classes[valid_bound:]
        }
