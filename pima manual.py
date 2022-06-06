#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:57:48 2022

@author: asad
"""


#importing the libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
import pandas

#fix the seed value

seed = 8
numpy.random.seed(seed)


def create_model():
#creating the model    
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    
    #compile model

    model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
    
    return model

#loading the dataset

dataset = numpy.loadtxt("pima-indians-diabetes.csv" , delimiter=",")

#split to input and output arrays

X = dataset[:,0:8]
Y = dataset[:,8]

#define k-fold

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

#create model using keras

model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10)

results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())