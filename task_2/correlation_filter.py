import pandas as pd


class CorrelationFilter:

    def __init__(self, corr_thrs=0.6):
        self.corr_thrs = corr_thrs

    def fit_data(self, data):

        pd_data_fr = pd.DataFrame(data=data)

        data_corr = pd_data_fr.corr()

        ids = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if data_corr.iloc[i, j] >= self.corr_thrs:
                    if ids[j]:
                        ids[j] = False
        selected_ids = np.where(ids == True)

        return selected_ids
