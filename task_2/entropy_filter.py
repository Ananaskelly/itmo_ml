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
        _, probs, uniq_vals = self.calc_entropy(x)

        for i, val in enumerate(uniq_vals):
            tmp_y = y[x == val].astype(float)
            # print(tmp_y.shape)
            res += probs[i] * self.calc_entropy(tmp_y)[0]

        return res

    def calc_entropy(self, x):
        un_val, counts = np.unique(x, return_counts=True)
        prb = counts / x.shape[0]
        entropy = -np.sum(prb * np.log2(prb))
        return entropy, prb, un_val

    def IG(self, y, x):
        H, _, _ = self.calc_entropy(y)
        H_cond = self.calc_cond_entropy(y, x)
        return H - H_cond
