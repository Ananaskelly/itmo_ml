from task_3.random_search import RandomSearch
import numpy as np


A = 10
"""
    Random search:
        Found minimum 0.001072436878928329 at point: [0.0007914934126316275, -0.0021861513251395515]


"""


def rastrigin(x):

    x = np.array(x)
    assert len(x.shape) == 1, 'Input array must be 1D!'
    n = x.shape[0]
    return A*n + np.sum([(el**2 - A * np.cos(2 * np.pi * el)) for el in x])


def run_random_search():

    search_engine = RandomSearch(rastrigin)

    bound = [-0.5, 0.5]
    n = 2
    bound_arr = np.tile(bound, (n, 1))

    best_f, best_val = search_engine.fit(bound_arr, 10000)

    print('Found minimum {} at point: {}'.format(best_f, best_val))


if __name__ == '__main__':
    run_random_search()
