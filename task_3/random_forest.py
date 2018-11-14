import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RandomForest:

    def __init__(self, n_est):
        self.n_estimators = n_est
        self.train_size = 3

        self.random_forest = []

        for i in range(self.n_estimators):
            self.random_forest.append(DecisionTreeRegressor(random_state=np.random.randint(0, 1000)))

    def fit(self, x, y):

        n_x, x_s = x.shape
        ids_range = np.arange(0, n_x, step=1)

        for i in range(self.n_estimators):
            ids = np.random.choice(a=ids_range, size=self.train_size)

            x_sub = x[ids]
            y_sub = y[ids]

            self.random_forest[i].fit(x_sub, y_sub)

    def predict(self, x, return_std=True):

        y = []
        for i in range(self.n_estimators):
            y.append(self.random_forest[i].predict(x))

        return np.mean(y), np.std(y)
