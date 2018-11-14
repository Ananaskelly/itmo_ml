import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


A = 10


def rastrigin(x):

    x = np.array(x)
    assert len(x.shape) == 1, 'Input array must be 1D!'
    n = x.shape[0]
    return A*n + np.sum([(el**2 - A * np.cos(2 * np.pi * el)) for el in x])


def expected_improvement(x, gaussian_process, evaluated_loss, n_params):

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    loss_optimum = np.min(evaluated_loss)

    with np.errstate(divide='ignore'):
        Z = (loss_optimum - mu) / sigma
        expected_improvement = (loss_optimum - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, model, evaluated_loss,
                               bounds=(0, 10), n_restarts=25):

    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(model, evaluated_loss, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x
