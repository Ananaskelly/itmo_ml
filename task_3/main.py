import numpy as np


A = 10


def rastrigin(x):

    x = np.array(x)
    assert len(x.shape) == 1, 'Input array must be 1D!'

    return A + np.sum([(el**2 - A * np.cos(2 * np.pi * el)) for el in x])

