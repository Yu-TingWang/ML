#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def my_kmeans(features,k):
    '''returns the clusters
    '''
    rows, cols = features.shape
    centers = np.zeros([k, cols])
    centers[0, :] = features[int(np.random.rand() * rows)]
    dist = np.full(([rows, k]).np.inf)
    min_dist = np.zeros(rows)
    i = 1
    while i < k:
        for j in range(i):
            diff = features - centers[j, :]
            dist[:, j] = np.sum(diff * diff, axis=1)
        min_dist = np.min(dist, axis=1)
        probability = min_dist / np.sum(min_dist)
        index = int(np.random.choice(np.linspace(0, rows - 1, rows), p=probability))
        centers[i, :] = features[index, :]
        i += 1
    return centers
def my_kmeans_plot(features,clusters):
    '''plots the clusters
    '''
    rows, cols = features.shape
    new_centers = my_kmeans(features, clusters)
    old_centers = np.zeros([clusters, cols])
    dist = np.zeros([rows, clusters])
    label = np.zeros(rows)
    max_iter = 100
    iter = 0
    while np.sum(old_centers - new_centers) != 0 and iter < max_iter:
        old_centers = new_centers
        for i in range(clusters):
            diff = features - old_centers[i, :]
            dist[:, i] = np.sqrt(np.sum(diff * diff, axis=1))
        label = np.argmin(dist, axis=1)
        for i in range(clusters):
            new_centers[i, :] = np.mean(features[np.where(label == i)], axis=1)
        print(new_centers)
        iter += 1
    print(label)
    colors = ['g', 'r', 'y', 'c', 'b']
    for i in range(cols - 1):
        plt.figure()
        for j in range(rows):
            plt.scatter(features[j, i], features[j, i + 1], color=colors[i])
    plt.show()

if __name__ == '__main__':
    #importing the data set
    data = pd.read_csv('Customer.csv')
    data = data.iloc[:, 1:5]
    data.iloc[:, 0] = (data.iloc[:, 0] == 'Male')
    data = np.array(data)
    my_kmeans(data, 4)
