# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    arr = features[targets == selected_class]
    return sum(arr) / arr.shape[0]


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    values = features[targets == selected_class]
    return np.cov(values, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    likelihoods = []
    class_likelihood = []
    for i in range(test_features.shape[0]):
        for class_index in range(len(classes)):
            class_likelihood.append(likelihood_of_class(test_features[i, :], means[class_index], covs[class_index]))
        likelihoods.append(class_likelihood)
        class_likelihood = []
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    results = np.zeros(likelihoods.shape[0])
    el = np.array(range(likelihoods.shape[1]))
    for i in range(likelihoods.shape[0]):
        results[i] = el[likelihoods[i][:] == max(likelihoods[i])]
    return results


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''


if __name__ == '__main__':
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.6)

    # print(mean_of_class(train_features, train_targets, 0))

    # print(covar_of_class(train_features, train_targets, 0))

    # class_mean = mean_of_class(train_features, train_targets, 0)
    # class_cov = covar_of_class(train_features, train_targets, 0)
    # print(likelihood_of_class(test_features[0, :], class_mean, class_cov))

    # print(maximum_likelihood(train_features, train_targets, test_features, classes))

    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    print(predict(likelihoods))
