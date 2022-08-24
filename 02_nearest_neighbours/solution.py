# Author: Gregorio Talevi
# Date: 18/08/2022
# Project: Nearest Neighbours
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points
from help import remove_one


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    sum = 0
    for i in range(x.shape[0]):
        sum += np.power(x[i]-y[i], 2)
    return np.sqrt(sum)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    return np.argsort(euclidian_distances(x, points))[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    votes = np.zeros(len(classes))
    for t in targets:
        for i in range(len(classes)):
            if t == classes[i]:
                votes[i] += 1
                break
    return np.argmax(votes)


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest_points = k_nearest(x, points, k)
    targets = []
    for i in nearest_points:
        targets.append(point_targets[i])
    return vote(targets, classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    res = np.zeros((points.shape[0]))
    for i in range(len(points)):
        res[i] = knn(points[i], remove_one(points, i), remove_one(point_targets, i), classes, k)
    return res


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == point_targets[i]:
            correct_predictions += 1
    return correct_predictions / len(predictions)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    confusion_matrix = np.zeros((len(classes), len(classes)))
    predictions = knn_predict(points, point_targets, classes, k)
    for i in range(len(predictions)):
        if predictions[i] == point_targets[i]:
            confusion_matrix[classes[int(predictions[i])]][int(predictions[i])] += 1
        else:
            confusion_matrix[classes[int(predictions[i])]][int(point_targets[i])] += 1
    return confusion_matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    best_n = 0
    accuracy = 0
    for k in range(1, len(points)-1, 1):
        new_accuracy = knn_accuracy(points, point_targets, classes, k)
        if new_accuracy > accuracy:
            accuracy = new_accuracy
            best_n = k
    return best_n


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    predictions = knn_predict(points, point_targets, classes, k)
    colors = ['yellow', 'purple', 'blue']
    for i in range(len(predictions)):
        [x, y] = points[i, :2]
        if predictions[i] == point_targets[i]:
            edge = 'green'
        else:
            edge = 'red'
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=edge, linewidths=2)

    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    total_weights_sum = 0
    weighted_votes = np.zeros(len(classes))
    for t in range(targets.shape[0]):
        for i in range(len(classes)):
            if targets[t] == classes[i]:
                weighted = 1/distances[t]  # TODO possibili divisioni per zero incontrate
                weighted_votes[i] += weighted
                total_weights_sum += weighted
                break
    for v in range(weighted_votes.shape[0]):
        if weighted_votes[v] != 0:
            weighted_votes[v] /= total_weights_sum  # TODO possibili divisioni per zero incontrate
    return np.argmax(weighted_votes)


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest_indexes = k_nearest(x, points, k)
    targets = []
    nearest_points = []
    for i in nearest_indexes:
        targets.append(point_targets[i])
        nearest_points.append(points[i])
    distances = euclidian_distances(x, np.array(nearest_points))
    return weighted_vote(np.array(targets), distances, classes)


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    res = np.zeros((points.shape[0]))
    for i in range(len(points)):
        res[i] = wknn(points[i], remove_one(points, i), remove_one(point_targets, i), classes, k)
    return res


def wknn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = wknn_predict(points, point_targets, classes, k)
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == point_targets[i]:
            correct_predictions += 1
    return correct_predictions / len(predictions)


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    accuracy = np.zeros(points.shape[0])
    w_accuracy = np.zeros(points.shape[0])
    for k in range(1, points.shape[0]):
        accuracy[k-1] = knn_accuracy(points, targets, classes, k)
        w_accuracy[k-1] = wknn_accuracy(points, targets, classes, k)
    plt.plot(range(0, len(targets), 1), accuracy)
    plt.plot(range(0, len(targets), 1), w_accuracy)
    plt.show()


# MAIN PART

if __name__ == '__main__':
    # d, t, classes = load_iris()
    # # plot_points(d, t)
    # x, points = d[0, :], d[1:, :]
    # x_target, point_targets = t[0], t[1:]

    # print(euclidian_distance(x, points[0]) == 0.5385164807134502)
    # print(euclidian_distance(x, points[50]) == 3.6166282640050254)
    # print(euclidian_distances(x, points)[0] == 0.5385164807134502)
    # print(k_nearest(x, points, 1)[0] == 16)
    # print(vote(np.array([0,0,1,2]), np.array([0,1,2])) == 0)
    # print(vote(np.array([1,1,1,1]), np.array([0,1])) == 1)
    # print(knn(x, points, point_targets, classes, 1) == 0)
    # print(knn(x, points, point_targets, classes, 5) == 0)
    # print(knn(x, points, point_targets, classes, 150) == 1)

    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
    # print(d_train)
    # print(t_train)
    # print(d_test)
    # print(t_test)
    # print(classes)

    # print(knn_predict(d_test, t_test, classes, 10)) # [2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 1 1 1 1 1 2 0 1 1 1]
    # print(knn_predict(d_test, t_test, classes, 5)) # [2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 2 1 1 2 1 2 0 1 1 2]
    # print(knn_accuracy(d_test, t_test, classes, 10) == 0.8275862068965517)
    # print(knn_accuracy(d_test, t_test, classes, 5) == 0.9310344827586207)
    # print(knn_confusion_matrix(d_test, t_test, classes, 10)) # [[ 6.  0.  0.] [ 0. 10.  4.] [ 0.  1.  8.]]
    # print(knn_confusion_matrix(d_test, t_test, classes, 20)) # [[ 0.  0.  0.] [ 6.  8.  1.] [ 0.  3. 11.]]
    # print(best_k(d_train, t_train, classes) == 9)
    # knn_plot_points(d, t, classes, 3)
    compare_knns(d, t, classes)


'''
## Answer to the Independent Part question ##

When we use a basic knn implementation, as k increases the classes that are more frequent in the dataset will have more
"weight" in the predictions. Ultimately, when k is equal to the number of points in the dataset, every prediction will
just output the class with the most points in the dataset. So this approach is very susceptible to the overall
balance of the initial dataset.

When we use a weighted knn implementation, instead, we give more weight the points that are actually near the new point
presented. So, even if we have a lot of points belonging to a specific class X, but very distant from the new point,
they will have a relatively weight in the final prediction, thus allowing for more coherent results. 
'''