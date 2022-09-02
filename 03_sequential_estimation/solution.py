# Author: Gregorio Talevi
# Date:
# Project: Sequential Estimation
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    np.random.seed(1234)
    cov = np.power(var,2)*np.identity(k)  # TODO NB! Non è questa la formula della covarianza!
    return np.random.multivariate_normal(mean, cov, n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    # TODO dobbiamo generalizzare l'algoritmo? Qui suppongo di avere solamente 1 nuovo input, non di più
    if n == 0:
        return mu
    return mu + (x-mu)/n


def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), 1)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i], data[i], i))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    error = 0
    for i in range(y.shape[0]):
        error += np.power(y[i]-y_hat[i],2)
    error /= y.shape[0]
    return error


def _plot_mean_square_error():
    data = gen_data(100, 3, np.array([0, 0, 0]), 1)
    errors = []
    estimates = [np.array([0, 0, 0])]
    actual_mean = np.mean(data, 0)
    for i in range(data.shape[0]):
        estimated_mean = update_sequence_mean(estimates[i], data[i], i+1)
        estimates.append(estimated_mean)
        errors.append(_square_error(actual_mean, estimated_mean))
    plt.plot(errors)
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    np.random.seed(1234)
    data = []
    cov = np.power(var,2)*np.identity(k)  # TODO NB! Non è questa la formula della covarianza!
    mean = start_mean
    for i in range(n):
        data.append(np.random.multivariate_normal(mean, cov, 1))
        # TODO trovare un modo per far evolvere la media
    return data


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    data = gen_changing_data(100, 3, np.array([0, 1, -1]), np.array([1,-1,0]), 1)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        tmp = update_sequence_mean(estimates[i], data[i], i)
        estimates.append(tmp)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


if __name__ == '__main__':
    # print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    # print(gen_data(5, 1, np.array([0.5]), 0.5))
    # X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    # scatter_3d_data(X)
    # bar_per_axis(X)
    # mean = np.mean(X, 0)
    # new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    # updated_mean = update_sequence_mean(mean, new_x, X.shape[0])
    # print(updated_mean)
    # _plot_sequence_estimate()
    _plot_mean_square_error()

    # print(_square_error(np.array([0, 1, -1]), np.array([0, 0.5, -0.5])))
    ...


# Answer to question 1.2: variance is definitely one of the 2 parameters to manipulate in order to get a more precise
# batch estimate. The other one it's the number of points we generate. The more points we get, the more precise the
# distribution will look like.
