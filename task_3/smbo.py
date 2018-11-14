import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize


class SMBO:

    def __init__(self, n_iter, loss_fnc, model, asc_fnc, sam_fnc):
        self.n_iter = n_iter
        self.loss_function = loss_fnc
        self.model = model
        self.asc_function = asc_fnc
        self.sample_next_params = sam_fnc
        self.eps = 1e-6

    def optimize(self, x0, n, bounds):

        x_list = []
        y_list = []

        if x0 is None:
            for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n, bounds.shape[0])):
                x_list.append(params)
                y_list.append(self.loss_function(params))
        else:
            for params in x0:
                x_list.append(params)
                y_list.append(self.loss_function(params))

        xp = np.array(x_list)
        yp = np.array(y_list)

        y_best = []

        for n in range(self.n_iter):

            self.model.fit(xp, yp)

            next_sample = self.sample_next_params(self.asc_function, self.model, yp, bounds=bounds, n_restarts=10)

            if np.any(np.abs(next_sample - xp) <= self.eps):
                next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

            scores = self.loss_function(next_sample)

            x_list.append(next_sample)
            y_list.append(scores)

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)

            y_best.append(np.min(yp))

        return xp, yp, y_best

