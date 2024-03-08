# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:32:45 2024

@author: Tran Thanh
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
import os

# Import helper functions
from utils import train_test_split, standardize, accuracy_score
from utils import mean_squared_error, Plot
from Decisiontree import ClassificationTree

def main():

    print ("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    clf.print_tree()

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, 
        title="Decision Tree", 
        accuracy=accuracy, 
        legend_labels=data.target_names)


if __name__ == "__main__":
    main()