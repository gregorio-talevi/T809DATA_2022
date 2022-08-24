# Author: Gregorio Talevi
# Date: 18/08/2022
# Project: 01_decision_trees
# Acknowledgements:
#

from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    samples_count = len(targets)
    class_probs = []

    if samples_count != 0:  # TODO controllare bene!
        for c in classes:
            class_count = 0
            for t in targets:
                if t == c:
                    class_count += 1
            class_probs.append(class_count / samples_count)

    return class_probs


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[features[:, split_feature_index] < theta, :]
    targets_1 = targets[features[:, split_feature_index] < theta]

    features_2 = features[features[:, split_feature_index] >= theta, :]
    targets_2 = targets[features[:, split_feature_index] >= theta]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    p = prior(targets, classes)
    sum = 0
    for i in p:
        sum += np.power(i, 2)
    return 0.5*(1-sum)


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]

    return (t1.shape[0]*g1 + t2.shape[0]*g2) / n


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1, t_2, classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        # for each list of values of a single feature pick the minimum and maximum values
        # and create num_tries linearly spaced values between them
        thetas = np.linspace(features[i].min(), features[i].max(), num_tries)  # TODO rivedere
        # iterate thresholds
        for theta in thetas:
            new_gini = total_gini_impurity(features, targets, classes, i, theta)
            if new_gini < best_gini:
                best_gini = new_gini
                best_dim = i
                best_theta = theta

    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets), (self.test_features, self.test_targets) = split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plot_tree(self.tree)
        plt.show()
        # plt.savefig("2_3_1", format="png")

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        tr_f, tr_t, ts_f, ts_t = self.train_features, self.train_targets, self.test_features, self.test_targets
        accuracy_series = []  # np.zeros(len(self.test_targets))

        for i in range(1, len(self.train_targets)):
            self.train_features, self.train_targets, self.test_features, self.test_targets = tr_f[:i], tr_t[:i], ts_f[:i], ts_t[:i]
            self.train()
            # accuracy_series[i-1] = self.accuracy()
            accuracy_series.append(self.accuracy())

        # accuracy_series.pop(0)  # TODO eliminare

        plt.plot(range(0, len(self.train_targets), 1), accuracy_series)  # TODO una volta elimintato il pop sopra rimettere il range a partire da 1
        plt.show()

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        confusion_matrix = np.zeros((len(classes),len(classes)))
        guesses = self.guess()
        for i in range(len(guesses)):
            if guesses[i] == self.test_targets[i]:
                confusion_matrix[classes[guesses[i]]][guesses[i]] += 1
            else:
            # elif guesses[i] != self.test_targets[i]:
                confusion_matrix[classes[guesses[i]]][self.test_targets[i]] += 1

        return confusion_matrix


# MAIN PART

if __name__ == '__main__':
    features, targets, classes = load_iris()

    print(brute_best_split(features, targets, classes, 30)) # TODO controllare bene! Non viene proprio il risultato del prof

    tree = IrisTreeTrainer(features, targets)
    tree.train()

    # print(tree.accuracy())
    # print(tree.guess())
    # print(tree.confusion_matrix())
    # tree.plot()
    # ------------------

    # features, targets, classes = load_iris()
    # dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
    # dt.plot_progress()

    # ------------------
    # self.train()
    # self.tree.predict(self.test_features)

