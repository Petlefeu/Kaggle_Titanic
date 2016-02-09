#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Kaggle Titanic """


import logging
import os
# import itertools as it
# from multiprocessing import Process
import numpy as np
import sklearn as sk
# import random
from pdb import set_trace as st
# import scipy as sp
# from scipy import signal
from sklearn import metrics, cross_validation, svm, linear_model, ensemble, cluster
from matplotlib import pyplot as plt
# from multiprocessing import Process, Queue
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

CSV_DIRECTORY = "passengers"
PICKLES_DIRECTORY = "pickles"
TRAIN_FILE = 'train.csv'
PASSENGERS = os.listdir('passengers')       # List of driver names ('1', '10', '100', ...)

NB_CPU = 8


######################################
## Pre-loading data into pickles    ##
######################################

def preload_into_pickles():
    """ Sauvegarde des informations en pickles """
    # Ne conserve que les entiers !
    passengers_infos = np.genfromtxt("%s/%s" % (CSV_DIRECTORY, TRAIN_FILE), skip_header=True, delimiter=',')
    np.save("%s/passengers.npy" % PICKLES_DIRECTORY, passengers_infos)

def read_from_pickles():
    """ Lecture des pointeurs """
    return np.load("%s/passengers.npy" % PICKLES_DIRECTORY)

######################################
##        Features computation      ##
######################################


def compute_features(passengers_infos):
    """ Cette fonction met en forme les attributs d'un individu pour qu'ils
    soient lisible par le classifier """

    passengers_survived = passengers_infos[:, 1]
    passenger_id = passengers_infos[:, 0]
    passenger_pclass = passengers_infos[:, 2]
    passenger_age = passengers_infos[:, 6]
    passenger_sibsp = passengers_infos[:, 7]
    passenger_parch = passengers_infos[:, 8]
    passenger_ticket = passengers_infos[:, 9]
    passenger_fare = passengers_infos[:, 10]

    features = [
        passenger_fare
    ]
    return np.array(features).T, passengers_survived


######################################
##       Training & testing         ##
######################################


# Trains a classifier for the given driver
def train(passengers_infos):
    """ Cette fonction entraine un classifier
    grace aux features de passengers_infos """


    passengers_features, passengers_survived = compute_features(passengers_infos)

    # Training
    # cls = sk.linear_model.LogisticRegression(C=1)
    # cls = sk.svm.SVC(C=0.01, probability=True)
    # cls = sk.ensemble.RandomForestClassifier(n_estimators=200, max_features=None)
    cls = sk.ensemble.AdaBoostClassifier()
    cls.fit(passengers_features, passengers_survived)

    return cls


#########################################################
##              Pre-loading data                       ##
#########################################################
# preload_into_pickles()

#########################################################
##    Computing final predictions for all drivers      ##
#########################################################

PASSENGERS_INFOS = read_from_pickles()

CLASSIFIER = train(PASSENGERS_INFOS)



#########################################################
##           Manual testing / feature creation         ##
#########################################################

T = []

for t in np.linspace(0, 100, num=500):
    T += [CLASSIFIER.predict_proba(t)[0][1]]

plt.figure()
plt.title('Survived')
plt.plot(np.linspace(0, 100, num=500), np.array(T), '.-')

plt.show()
