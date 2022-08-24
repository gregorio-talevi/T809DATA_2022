# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

import help
from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    sum=0
    for i in range(x.shape[0]):
        sum+=np.power(x[i]-y[i],2)
    return np.sqrt(sum)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i]=euclidian_distance(x,points[i])
    return distances

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    return np.argsort(euclidian_distances(x,points))[:k]




def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    value=0
    map ={}
    max=-100
    for c in classes:
        map[c]=0
    for i,element in enumerate(targets):
        map[element]+=1
        if(map[element]> max and (element in classes)):
            max= map[element]
            value = element
    #print(targets)
    return value
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
    distances=k_nearest(x,points,k)
    targets=[]
    for element in distances:
        targets.append(point_targets[element])
    return vote(targets,classes)

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    result =[]
    for i in range(len(points)):
        X=help.remove_one(points,i)
        y=help.remove_one(point_targets,i)
        result.append(knn(points[i],X,y,classes,k))
    return np.array(result)



def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    cont =0
    result = knn_predict(points,point_targets,classes,k)
    for i in range(len(result)):
        if result[i]==point_targets[i]:
            cont+=1
    return cont/len(result)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    y_pred = knn_predict(points,point_targets,classes,k)
    confusion_matrix= [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(y_pred)):
        confusion_matrix[y_pred[i]][point_targets[i]]+=1
    # for line in confusion_matrix:
        # print ('  '.join(map(str, line)))
    return np.asmatrix(confusion_matrix)

def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    best= -float("inf")
    position= 0
    for i in range(1,len(point_targets)):
        accurancy_i=knn_accuracy(points,point_targets,classes,i)
        if accurancy_i> best:
            best= accurancy_i
            position=i

    return position

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colors = ['green', 'red']
    result =knn_predict(points,point_targets,classes,k)

    for i in range(0,point_targets.shape[0]):

        [x, y] = points[i,:2]
        if(point_targets[i]== result[i]):
            plt.scatter(x, y, c=colors[0], edgecolors='green',
            linewidths=2)
        else:
            plt.scatter(x, y, c=colors[1], edgecolors='red',
            linewidths=2)
    plt.title('Prediction: Green=Y ,Red=N')
    plt.savefig("2_5_1.png")
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
    # Remove if you don't go for independent section
    map= {c:0 for c in classes}
    max= -float("inf")
    pos=0
    for i,value in enumerate(distances):
        map[targets[i]]+= 1/value
    for c in classes:
        if (map[c] > max):
            max=map[c]
            pos=c
    # print(map)
    return pos

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
    # Remove if you don't go for independent section
    distances_nearest=k_nearest(x,points,k)
    distances_values=[]
    for element in distances_nearest:
        distances_values.append(euclidian_distance(x,points[element]))
    targets=[]
    for element in distances_nearest:
        targets.append(point_targets[element])
    return weighted_vote(np.asarray(targets[:k]),np.asarray(distances_values[:k]),classes)



def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    result=[]
    for i in range(len(points)):
        X=help.remove_one(points,i)
        y=help.remove_one(point_targets,i)
        result.append(wknn(points[i],X,y,classes,k))


    return np.asarray(result)


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    accurancy_knn=[]
    accurancy_wknn=[]
    for i in range(1,points.shape[0]):
        accurancy_knn.append(knn_accuracy(points,targets,classes,i))
        accurancy_wknn.append(wknn_accuracy(points,targets,classes,i))

    plt.plot(range(1,points.shape[0]),accurancy_knn)
    plt.plot(range(1,points.shape[0]),accurancy_wknn)
    plt.show()

def wknn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    cont =0
    result = wknn_predict(points,point_targets,classes,k)
    for i in range(len(result)):
        if result[i]==point_targets[i]:
            cont+=1
    return cont/len(result)

# d, t, classes = load_iris()
# x, points = d[0,:], d[1:, :]
# x_target, point_targets = t[0], t[1:]

#print(euclidian_distance(x, points[0]))
#print(euclidian_distance(x, points[50]))
#print(euclidian_distances(x, points))
'''

print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
print(vote(np.array([1,1,1,1]), np.array([0,1])))

print(knn(x, points, point_targets, classes, 1))
print(knn(x, points, point_targets, classes, 5))
print(knn(x, points, point_targets, classes, 150))
'''

# print(k_nearest(x, points, 1))
# print(k_nearest(x, points, 3))
# print(knn(x, points, point_targets, classes, 1))
# print(knn(x, points, point_targets, classes, 5))
# print(knn(x, points, point_targets, classes, 150))
# d, t, classes = load_iris()
# (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
#
# r1=[2,2,2,2,0,1,0,1,1,0,1,2,1,2,2,0,1,0,2,1,1,1,1,1,2,0,1,1,1]
# l=compare_knns(d_test, t_test, classes)
# print(l)
'''
l=knn_predict2(d_test, t_test, classes, 10)

print(l)
print(r1[22])
'''
#knn_plot_points(d, t, classes, 3)
#print(d_test)

#l=knn_predict(d_test, t_test, classes, 10)
#print(l)
#l=compare_knns(d_test, t_test, classes)
#print(l)
#compare_knns(d_test,t_test,classes)


#print(t_test)
'''
for i in range(l.shape[0]):
    if r1[i]!=l[i]:
        print("False1")
        print(i)
        print(r1[i])
        print(l[i])

l=knn_predict(d_test, t_test, classes, 10)
r1=[2,2,2,2,0,1,0,1,1,0,1,2,1,2,2,0,1,0,2,1,1,1,1,1,2,0,1,1,1]
for i in range(l.shape[0]):
    if r1[i]!=l[i]:
        print("False1")
        print(i)
        print(r1[i])
        print(l[i])
l=knn_predict(d_test, t_test, classes, 5)
r2=[2,2,2,2,0,1,0,1,1,0,1,2,1,2,2,0,1,0,2,2,1,1,2,1,2,0,1,1,2]
for i in range(l.shape[0]):
    if r2[i]!=l[i]:
        print("False")
'''
#print(knn_accuracy(d_test, t_test, classes, 10))
#print(knn_accuracy(d_test, t_test, classes, 5))
#knn_confusion_matrix(d_test, t_test, classes, 10)
#knn_confusion_matrix(d_test, t_test, classes, 20)
#print(best_k(d_train, t_train, classes))


if __name__ == '__main__':
    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

    compare_knns(d, t, classes)

