import numpy as np


class RandomSearch:

    def __init__(self, objective_func):
        self.obj_func = objective_func

    def fit(self, var_bounds, max_iter):

        num_v, _ = var_bounds.shape
        best_vals = np.zeros(num_v)
        best_f = None

        for i in range(max_iter):
            curr_v = [np.random.uniform(var_bounds[j, 0], var_bounds[j, 1]) for j in range(num_v)]

            f_v = self.obj_func(curr_v)
            if best_f is None or best_f > f_v:
                best_f = f_v
                best_vals = curr_v

        return best_f, best_vals




