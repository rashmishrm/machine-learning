#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import sys
from time import time

from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
classifier = GaussianNB()
classifier.fit(features_train, labels_train)
results = classifier.predict(features_test);
score=accuracy_score(results, labels_test)
print score
