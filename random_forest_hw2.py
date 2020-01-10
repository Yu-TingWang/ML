# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    train_data_x = train_data.iloc[:,:-1].values
    train_data_y = train_data.iloc[:,11].values
    test_data_x = test_data.iloc[:,:-1].values
    test_data_y = test_data.iloc[:,11].values


    test_data = test_data.dropna()
    forest = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)
    forest.fit(train_data_x, train_data_y)

    predict_y = forest.predict(test_data_x)

    (truen, falsep, falsen, truep) = confusion_matrix(test_data_y, predict_y).ravel()
