# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data


def my_pca(data_matrix, k):
    mean = np.mean(data_matrix, axis = 0)
    centers = data_matrix - mean
    cov = np.cov(centers.T)
    values, vectors = eig(cov)
    projected = centers.dot(vectors)
    return projected[:, 0:k]


def my_pca_plot(low_dim_matrix):
    rows, cols = low_dim_matrix.shape
    i=0
    while i<cols-1:
        plt.figure()
        j=0
        while j<rows:
            plt.scatter(low_dim_matrix[j, i], low_dim_matrix[j, i + 1])
            j+=1
        i+=1
    plt.show()


