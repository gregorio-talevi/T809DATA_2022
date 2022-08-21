import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    res = []
    for elem in x:
        res.extend((1/np.sqrt(2*np.pi * np.power(sigma,2))) * np.exp(-np.power((elem-mu),2)/(2*np.power(sigma,2))))
    return np.ndarray(res)
    # return (1 / np.sqrt(2 * np.pi * np.power(sigma, 2))) * np.exp(-np.power((x - mu), 2) / (2 * np.power(sigma, 2)))


if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    print(normal(np.ndarray([0,1,0]), 1, 0))
    # print(normal(3, 1, 5))
# print(normal(np.array([-1,0,1]), 1, 0))
