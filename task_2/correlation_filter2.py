import pandas as pd
import numpy as np


class CorrelationFilter:

    def __init__(self, corr_thrs=0.4, num_feats=50):
        self.corr_thrs = corr_thrs
        self.num_feats = num_feats

    def fit_data(self, data, labels):

        _, feats_num = data.shape

        all_val = []

        for i in range(feats_num):
            all_val.append(np.corrcoef(data[:, i], labels)[0, 1])

        all_val = np.abs(np.array(all_val))

        selected_ids = np.where(all_val > self.corr_thrs)[0]

        return selected_ids[:self.num_feats]
