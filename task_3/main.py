from task_3.random_search import RandomSearch
import numpy as np


A = 10
"""
    Random search:
        Found minimum 0.057947588885156165 at point: [0.005904906746808436, 0.01604490908983891]


"""


def rastrigin(x):

    x = np.array(x)
    assert len(x.shape) == 1, 'Input array must be 1D!'
    n = x.shape[0]
    return A*n + np.sum([(el**2 - A * np.cos(2 * np.pi * el)) for el in x])


def run_random_search():

    search_engine = RandomSearch(rastrigin)

    bound = [-5.12, 5.12]
    n = 2
    bound_arr = np.tile(bound, (n, 1))

    best_f, best_val = search_engine.fit(bound_arr, 10000)

    print('Found minimum {} at point: {}'.format(best_f, best_val))


if __name__ == '__main__':
    run_random_search()
