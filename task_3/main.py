import numpy as np
import sklearn.gaussian_process as gp

from task_3.random_search import RandomSearch
from task_3.utils import rastrigin
from task_3.utils import expected_improvement
from task_3.utils import sample_next_hyperparameter

from task_3.smbo import SMBO
from task_3.random_forest import RandomForest

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


A = 10
"""
    Random search:
        Found minimum 0.057947588885156165 at point: [0.005904906746808436, 0.01604490908983891]


"""


def run_random_search():

    search_engine = RandomSearch(rastrigin)

    bound = [-5.12, 5.12]
    n = 2
    bound_arr = np.tile(bound, (n, 1))

    best_f, best_val, val_arr, best_y = search_engine.fit(bound_arr, 15)

    print('Found minimum {} at point: {}'.format(best_f, best_val))

    return best_y


def run_gauss_smbo():
    bound = [-5.12, 5.12]
    n = 2

    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                        alpha=1e-5,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)

    bounds_arr = np.tile(bound, (n, 1))
    smbo_engine = SMBO(n_iter=15, loss_fnc=rastrigin, model=model, asc_fnc=expected_improvement,
                       sam_fnc=sample_next_hyperparameter)

    x, y, y_best = smbo_engine.optimize(x0=None, n=5, bounds=bounds_arr)

    win_idx = np.argmin(y)

    print('Found minimum {} at point: {}'.format(y[win_idx], x[win_idx]))

    return y_best


def run_RF_smbo():
    bound = [-5.12, 5.12]
    n = 2

    model = RandomForest(n_est=10)

    bounds_arr = np.tile(bound, (n, 1))
    smbo_engine = SMBO(n_iter=15, loss_fnc=rastrigin, model=model, asc_fnc=expected_improvement,
                       sam_fnc=sample_next_hyperparameter)

    x, y, y_best = smbo_engine.optimize(x0=None, n=5, bounds=bounds_arr)

    win_idx = np.argmin(y)

    print('Found minimum {} at point: {}'.format(y[win_idx], x[win_idx]))

    return y_best

if __name__ == '__main__':

    rnd_v_all = []
    smbo_1_all = []
    smbo_2_all = []

    for i in range(5):
        rnd_v_all.append(run_random_search())
        smbo_1_all.append(run_gauss_smbo())
        smbo_2_all.append(run_RF_smbo())

    rnd_v_mean = np.mean(np.stack(rnd_v_all), axis=0)
    smbo_1_mean = np.mean(np.stack(smbo_1_all), axis=0)
    smbo_2_mean = np.mean(np.stack(smbo_2_all), axis=0)

    n = len(rnd_v_mean)
    plt.plot(np.arange(0, n, step=1), rnd_v_mean, 'r', np.arange(0, n, step=1), smbo_1_mean, 'g',
             np.arange(0, n, step=1), smbo_2_mean, 'b')

    red_patch = mpatches.Patch(color='red', label='random search')
    green_patch = mpatches.Patch(color='green', label='smbo (gaussian process)')
    blue_patch = mpatches.Patch(color='blue', label='smbo (random forest)')
    plt.legend(handles=[red_patch, green_patch, blue_patch])

    plt.grid()
    plt.xlabel('n_iter')
    plt.ylabel('func_val')
    plt.savefig('result.jpg')

