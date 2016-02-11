#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Kaggle Titanic """


import logging
import os
# import itertools as it
# from multiprocessing import Process
import numpy as np
import sklearn as sk
import random
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

PERCENT_TRAINING = 80


NB_CPU = 8


######################################
## Pre-loading data into pickles    ##
######################################

def preload_into_pickles():
    """ Sauvegarde des informations en pickles """
    # Ne conserve que les entiers !
    passengers_infos = np.genfromtxt(
        "%s/%s" % (CSV_DIRECTORY, TRAIN_FILE),
        skip_header=True,
        delimiter=',',
        # usecols=(0, 1, 2, 6, 7, 8, 9))
        dtype=[('f0', 'i8'), ('f1', 'i8'), ('f2', 'i8'), ('f3', 'O'), ('f4', 'O'), ('f5', 'O'), ('f6', 'i8'), ('f7', 'i8'), ('f8', 'i8'), ('f9', 'O'), ('f10', 'f8'), ('f11', 'O'), ('f12', 'O')])
    np.save("%s/passengers.npy" % PICKLES_DIRECTORY, np.array(passengers_infos.tolist()))

def read_from_pickles():
    """ Lecture des pointeurs """
    return np.load("%s/passengers.npy" % PICKLES_DIRECTORY)

######################################
##        Features computation      ##
######################################

def sex_to_int(passengers_infos_sex):
    """ Cette fonction transforme le sexe en int :
    male = 0
    female = 1 """
    passengers_infos_sex[passengers_infos_sex == 'male'] = 0
    passengers_infos_sex[passengers_infos_sex == 'female'] = 1
    return passengers_infos_sex

def get_missing_age(passengers_infos_age):
    """ Cette fonction cree un vecteur
    age -1 => 1
    sinon  => 0 """
    passenger_missing_age = np.zeros(len(passengers_infos_age))

    for var, age in enumerate(passengers_infos_age):
        if int(age) == -1:
            passenger_missing_age[var] = 1
    return passenger_missing_age

def compute_features(passengers_infos):
    """ Cette fonction met en forme les attributs d'un individu pour qu'ils
    soient lisible par le classifier """

    passengers_survived = passengers_infos[:, 1]
    passenger_id = passengers_infos[:, 0]
    passenger_pclass = passengers_infos[:, 2]

    passenger_sex = sex_to_int(passengers_infos[:, 5])
    passenger_age = passengers_infos[:, 6]
    passenger_sibsp = passengers_infos[:, 7]
    passenger_parch = passengers_infos[:, 8]
    passenger_ticket = passengers_infos[:, 9]
    passenger_fare = passengers_infos[:, 10]
    passenger_missing_age = get_missing_age(passenger_age)


    features = [
        passenger_fare,
        passenger_sex,
        passenger_missing_age,
        passenger_pclass,
        passenger_sibsp
    ]
    return np.array(features).T.astype(float), passengers_survived.astype(int)


######################################
##       Training & testing         ##
######################################


# Trains a classifier for the given driver
def train(passengers_infos):
    """ Cette fonction entraine un classifier
    grace aux features de passengers_infos """

    passengers_features, passengers_survived = compute_features(passengers_infos)

    # Training

    # cls = sk.linear_model.LogisticRegression(C=0.01)
    # C1 69%, C100 70%, C0.01 60%

    # cls = sk.svm.SVC(C=0.01, probability=True)
    # C1 66%, C100 67%, C0.01 63%

    cls = sk.ensemble.RandomForestClassifier(n_estimators=200, max_features=None)
    # 200 75%, 20 75%, 1000 75%

    # cls = sk.ensemble.AdaBoostClassifier(n_estimators=200)
    # 50%

    # cls = sk.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    # 100_1_1_0 70%

    # cls = sk.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    # 68%

    # cls = sk.neighbors.KNeighborsClassifier(n_neighbors=1)
    # 7=68% 5=70% 1=75%(plus rapide que RFC)

    cls.fit(passengers_features, passengers_survived)

    return cls

def verify(classifier, passengers_infos):
    """ Vérification des prédictions d'un classifier
    sur un set de test """

    passengers_features, passengers_survived = compute_features(passengers_infos)

    passengers_survived_prediction = classifier.predict_proba(passengers_features)[:, 1]
    # passengers_survived_prediction = classifier.predict(passengers_features)

    return 1 - np.sum(abs(passengers_survived.astype(int) - passengers_survived_prediction.astype(float)))/len(passengers_survived)


#########################################################
##              Pre-loading data                       ##
#########################################################
# preload_into_pickles()

#########################################################
##    Computing final predictions for all passengers   ##
#########################################################

PASSENGERS_INFOS = read_from_pickles()

PROBA = 0
NB_LOOP = 1
for i in range(NB_LOOP):
    np.random.shuffle(PASSENGERS_INFOS)
    NB_TRAINING_PASSENGERS = len(PASSENGERS_INFOS)*PERCENT_TRAINING/100
    TRAINIG_SAMPLE = PASSENGERS_INFOS[:NB_TRAINING_PASSENGERS]
    TESTING_SAMPLE = PASSENGERS_INFOS[NB_TRAINING_PASSENGERS:]

    CLASSIFIER = train(TRAINIG_SAMPLE)
    PROBA += verify(CLASSIFIER, TESTING_SAMPLE)
    print i+1

print PROBA/NB_LOOP

print CLASSIFIER.feature_importances_

#########################################################
##           Manual testing / feature creation         ##
#########################################################

# T_HOMMES = []
# T_FEMMES = []

# for t in np.linspace(0, 100, num=500):
#     T_HOMMES += [CLASSIFIER.predict_proba([t, 0, 1, 3, 0])[0][1]]
#     T_FEMMES += [CLASSIFIER.predict_proba([t, 1, 1, 3, 0])[0][1]]

# plt.figure()
# plt.title('Survived')
# plt.axis([0, 100, 0, 1])
# plt.plot(np.linspace(0, 100, num=500), np.array(T_HOMMES), '.-', color='blue')
# plt.plot(np.linspace(0, 100, num=500), np.array(T_FEMMES), '.-', color='red')

# plt.show()
