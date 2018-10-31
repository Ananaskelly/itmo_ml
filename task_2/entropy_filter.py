import numpy as np


class EntropyFilter:

    def __init__(self, n=50):
        self.n = n

    def fit_data(self, x, y):

        return self.sort_feats(x, y)[:50]

    def sort_feats(self, x, y):
        IG_vals = self.calc_IG(x, y)
        ids = np.argsort(IG_vals)[::-1]

        return ids

    def calc_IG(self, x, y):

        IG_vals = []
        for i in range(x.shape[1]):
            IG_vals.append(self.IG(y, x[:, i]))

        return np.array(IG_vals)

    def calc_cond_entropy(self, y, x):
        res = 0
        probs, uniq_vals = self.get_probs(x)
        for i, val in enumerate(uniq_vals):
            curr_y = y[x == val].astype(float)
            res += probs[i] * self.calc_entropy(curr_y)

        return res

    def get_probs(self, x):
        un_val, counts = np.unique(x, return_counts=True)
        prb = counts / x.shape[0]

        return prb, un_val

    def calc_entropy(self, x):
        prb, un_val = self.get_probs(x)
        entropy = -np.sum(prb * np.log2(prb))
        return entropy

    def IG(self, y, x):
        H = self.calc_entropy(x)
        H_cond = self.calc_cond_entropy(x, y)
        return H - H_cond
