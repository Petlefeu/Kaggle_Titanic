#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Kaggle Titanic """


import logging
# import os
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

PERCENT_TRAINING = 80

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
        dtype=[('f0', 'i8'), ('f1', 'i8'), ('f2', 'i8'), ('f3', 'O'), ('f4', 'O'), ('f5', 'O'), ('f6', 'f8'), ('f7', 'i8'), ('f8', 'i8'), ('f9', 'O'), ('f10', 'f8'), ('f11', 'O'), ('f12', 'O')])
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

def embarked_to_int(passengers_infos_embarked):
    """ Cette fonction transforme le lieu d'embarcation en int :
    'S' = 0
    'Q' = 1
    'C' = 2
    None = 0 """
    passengers_infos_embarked[passengers_infos_embarked == 'S'] = 0
    passengers_infos_embarked[passengers_infos_embarked == 'Q'] = 1
    passengers_infos_embarked[passengers_infos_embarked == 'C'] = 2
    passengers_infos_embarked[passengers_infos_embarked == ''] = 0
    return passengers_infos_embarked

def name_type_to_int(passengers_infos_name):
    """ Cette fonction transforme les civilites en int :
    'Mr.', 'Mrs.', 'Mme.', 'Miss.', 'Mlle.', 'Ms.': 0
    'Col.', 'Capt.', 'Doc.', 'Dr.', 'Rev.', 'Master.', 'Major.', 'Don.', 'Jonkheer.', 'Countess.' : 1 """
    passenger_info_civilite = np.ones(len(passengers_infos_name))
    for var, passenger in enumerate(passengers_infos_name):
        if 'Mr.' in passenger:
            passenger_info_civilite[var] = 0
        if 'Mrs.' in passenger:
            passenger_info_civilite[var] = 0
        if 'Mme.' in passenger:
            passenger_info_civilite[var] = 0
        if 'Miss.' in passenger:
            passenger_info_civilite[var] = 0
        if 'Mlle.' in passenger:
            passenger_info_civilite[var] = 0
        if 'Ms.' in passenger:
            passenger_info_civilite[var] = 0
    return passenger_info_civilite

def fill_missing_age_mean(passengers_infos_age):
    """ Cette fonction remplasse les ages manquant """
    mean_age = np.nanmean(passengers_infos_age)
    passengers_infos_age_complete = np.zeros(len(passengers_infos_age))
    for var, age in enumerate(passengers_infos_age):
        if np.isnan(age):
            passengers_infos_age_complete[var] = mean_age
        else:
            passengers_infos_age_complete[var] = age
    return passengers_infos_age_complete

def fill_missing_age_median(passengers_infos_age):
    """ Cette fonction remplasse les ages manquant """
    median_age = np.nanmedian(passengers_infos_age)
    passengers_infos_age_complete = np.zeros(len(passengers_infos_age))
    for var, age in enumerate(passengers_infos_age):
        if np.isnan(age):
            passengers_infos_age_complete[var] = median_age
        else:
            passengers_infos_age_complete[var] = age
    return passengers_infos_age_complete

def get_missing_age(passengers_infos_age):
    """ Cette fonction cree un vecteur
    age -1 => 1
    sinon  => 0 """
    passenger_missing_age = np.zeros(len(passengers_infos_age))

    for var, age in enumerate(passengers_infos_age):
        if int(age) == -1:
            passenger_missing_age[var] = 1
    return passenger_missing_age

# def ticket_to_int(passengers_infos_ticket):
#     """ Cette fonction transforme le ticket en int :
#     float => 1
#     sinon => 0 """
#     passenger_float_ticket = np.zeros(len(passengers_infos_ticket))

#     for var, ticket in enumerate(passengers_infos_ticket):
#         try:
#             ticket.astype(float)
#             passenger_float_ticket[var] = 1
#         except ValueError:
#             passenger_float_ticket[var] = 0
#     return passenger_float_ticket

def deck_to_int(passengers_infos_cabin):
    """ Cette fonction trie par deck, de A à G et T ou O(ther)"""
    passengers_decks = np.zeros(len(passengers_infos_cabin))
    for var, cabin in enumerate(passengers_infos_cabin):
        if 'A' in cabin:
            passengers_decks[var] = 1
        if 'B' in cabin:
            passengers_decks[var] = 2
        if 'C' in cabin:
            passengers_decks[var] = 3
        if 'D' in cabin:
            passengers_decks[var] = 4
        if 'E' in cabin:
            passengers_decks[var] = 5
        if 'F' in cabin:
            passengers_decks[var] = 6
        if 'G' in cabin:
            passengers_decks[var] = 7
        if 'T' in cabin:
            passengers_decks[var] = 8
    return passengers_decks

def fare_per_person(passengers_infos):
    """ Cette fonction divise le prix du billet par le nombre
    de membre de la famille. En cas de 0 :
    Classe 1 : 30£
    Classe 2 : 12£
    Classe 3 : 5£ (3£ à 8£) """
    passengers_fare_alone = np.zeros(len(passengers_infos))
    for var, fare in enumerate(passengers_infos[:, (0, 2, 7, 8, 10)].astype(float)):
        fare_alone = fare[4]/(fare[2]+fare[3]+1)
        if fare_alone == 0 and fare[1] == 1:
            fare_alone = 30
        elif fare_alone == 0 and fare[1] == 2:
            fare_alone = 12
        elif fare_alone == 0 and fare[1] == 3:
            fare_alone = 5
        passengers_fare_alone[var] = fare_alone
    return passengers_fare_alone


def compute_features(passengers_infos):
    """ Cette fonction met en forme les attributs d'un individu pour qu'ils
    soient lisibles par le classifier """

    passengers_survived = passengers_infos[:, 1]

    passengers_pclass = passengers_infos[:, 2]
    passengers_sibsp = passengers_infos[:, 7]
    passengers_parch = passengers_infos[:, 8]
    passengers_sex = sex_to_int(passengers_infos[:, 5])
    passengers_deck = deck_to_int(passengers_infos[:, 11])
    passengers_int_embarked = embarked_to_int(passengers_infos[:, 12])
    passengers_int_civilite = name_type_to_int(passengers_infos[:, 4])
    passengers_infos_age_complete_mean = fill_missing_age_mean(passengers_infos[:, 6].astype(float))
    passengers_infos_age_complete_median = fill_missing_age_median(passengers_infos[:, 6].astype(float))
    passengers_fare_alone = fare_per_person(passengers_infos)

    # Non utilisées
    # passenger_id = passengers_infos[:, 0]
    # passengers_fare = passengers_infos[:, 10]
    # passengers_ticket = passengers_infos[:, 9]
    # passengers_float_ticket = ticket_to_int(passengers_ticket)

    features = [
        passengers_fare_alone,
        passengers_sex,
        passengers_infos_age_complete_mean,
        passengers_infos_age_complete_median,
        passengers_pclass,
        passengers_sibsp,
        passengers_deck,
        passengers_int_civilite,
        passengers_int_embarked,
        passengers_parch,
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

    # cls = sk.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0)
    # 100_1_1_0 75%

    # cls = sk.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    # 68%

    # cls = sk.neighbors.KNeighborsClassifier(n_neighbors=1)
    # 7=68% 5=70% 1=75%(plus rapide que RFC)

    cls.fit(passengers_features, passengers_survived)

    return cls

# Trains a classifier for the given driver
def train_all(passengers_infos):
    """ Cette fonction entraine tous les classifiers
    grace aux features de passengers_infos """

    passengers_features, passengers_survived = compute_features(passengers_infos)

    # Training

    cls1 = sk.linear_model.LogisticRegression(C=0.01).fit(passengers_features, passengers_survived)
    # C1 69%, C100 70%, C0.01 60%

    cls2 = sk.svm.SVC(C=0.01, probability=True).fit(passengers_features, passengers_survived)
    # C1 66%, C100 67%, C0.01 63%

    cls3 = sk.ensemble.RandomForestClassifier(n_estimators=200, max_features=None).fit(passengers_features, passengers_survived)
    # 200 75%, 20 75%, 1000 75%

    cls4 = sk.ensemble.AdaBoostClassifier(n_estimators=200).fit(passengers_features, passengers_survived)
    # 50%

    cls5 = sk.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(passengers_features, passengers_survived)
    # 100_1_1_0 70%

    # cls6 = sk.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(passengers_features, passengers_survived)
    # 68%

    cls7 = sk.neighbors.KNeighborsClassifier(n_neighbors=1).fit(passengers_features, passengers_survived)
    # 7=68% 5=70% 1=75%(plus rapide que RFC)

    return cls1, cls2, cls3, cls4, cls5, cls7


def verify(classifier, passengers_infos):
    """ Vérification des prédictions d'un classifier
    sur un set de test """

    passengers_features, passengers_survived = compute_features(passengers_infos)

    passengers_survived_prediction = classifier.predict_proba(passengers_features)[:, 1]
    # passengers_survived_prediction = classifier.predict(passengers_features)

    return 1 - np.sum(abs(passengers_survived.astype(int) - passengers_survived_prediction.astype(float)))/len(passengers_survived)

def verify_all(classifiers, passengers_infos):
    """ Vérification des prédictions des classifiers
    sur un set de test """

    passengers_features, passengers_survived = compute_features(passengers_infos)
    for classifier in classifiers:
        passengers_survived_prediction = classifier.predict_proba(passengers_features)[:, 1]
        print 1 - np.sum(abs(passengers_survived.astype(int) - passengers_survived_prediction.astype(float)))/len(passengers_survived)


#########################################################
##              Pre-loading data                       ##
#########################################################
# preload_into_pickles()

#########################################################
##    Computing final predictions for all passengers   ##
#########################################################

PASSENGERS_INFOS = read_from_pickles()

PROBA = 0
NB_LOOP = 10
for i in range(NB_LOOP):
    np.random.shuffle(PASSENGERS_INFOS)
    NB_TRAINING_PASSENGERS = len(PASSENGERS_INFOS)*PERCENT_TRAINING/100
    TRAINIG_SAMPLE = PASSENGERS_INFOS[:NB_TRAINING_PASSENGERS]
    TESTING_SAMPLE = PASSENGERS_INFOS[NB_TRAINING_PASSENGERS:]

    CLASSIFIER = train(TRAINIG_SAMPLE)
    # CLASSIFIERS = train_all(TRAINIG_SAMPLE)
    PROBA += verify(CLASSIFIER, TESTING_SAMPLE)
    # verify_all(CLASSIFIERS, TESTING_SAMPLE)
    print i+1

print PROBA/NB_LOOP

print CLASSIFIER.feature_importances_*len(CLASSIFIER.feature_importances_)
print CLASSIFIER.feature_importances_

#########################################################
##           Manual testing / feature creation         ##
#########################################################

# T_HOMMES = []
# T_FEMMES = []

# for t in np.linspace(0, 100, num=500):
#     T_HOMMES += [CLASSIFIER.predict_proba([t, 0, 1, 1, 1, 1, 1, 1, 1])[0][1]]
#     T_FEMMES += [CLASSIFIER.predict_proba([t, 1, 1, 1, 1, 1, 1, 1, 1])[0][1]]

# plt.figure()
# plt.title('Survived')
# plt.axis([0, 100, 0, 1])
# plt.plot(np.linspace(0, 100, num=500), np.array(T_HOMMES), '.-', color='blue')
# plt.plot(np.linspace(0, 100, num=500), np.array(T_FEMMES), '.-', color='red')

# plt.show()
