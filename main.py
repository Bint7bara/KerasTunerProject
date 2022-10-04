import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#import matplotlib.pyplot as plt
import numpy as np
from math import fabs

seed = 7
# load pima indians dataset
dataset = np.loadtxt("NewGencode4DLTraining.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:5]
y = dataset[:,5]

def build_model(hp):
    input_shape = (X.shape[1],)

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape)) # shape=(32, 32, 3)))
    print("Adding Input Layer with input_shape:",input_shape)

    model.add(layers.Flatten())
    print("Flattening Input Layer")

    nLayers = hp.Int('num_layers', 1, 5)  # max range is INCLUSIVE
    print("max range for num_layers is INCLUSIVE:", nLayers)
    for i in range(nLayers):             # loop from 0 to nLayers-1 Inclusive to add hidden layers
        hp_min_value=10 + nLayers*10 - 20*fabs(i-nLayers/2) # constant + scale - bulge
        hp_max_value=20 + int(100/nLayers) + nLayers*50 - 100*fabs(i-nLayers/2) # constant + underdog + scale - bulge
        hpunits=hp.Int('units_' + str(i),
                                            min_value=hp_min_value,
								#10+nLayers*10-20*fabs(i-nLayers/2),
                                            max_value=hp_max_value,
								#20+nLayers*50-100*fabs(i-nLayers/2),
                                            step=10)
        model.add(layers.Dense(units=hpunits,
                               activation='relu'))
        #print("Adding layer ",i," in range: ", 40-20*fabs(i-nLayers/2), " to ", 200-100*fabs(i-nLayers/2))
        print("Adding layer ",i," in range: ", hp_min_value, " < ", hpunits, " < ", hp_max_value)
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=3,
    directory='project',
    project_name='Gencode7')

tuner.search_space_summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuner.search(X_train, y_train,
             epochs=150,
             validation_data=(X_test, y_test))

tuner.results_summary()









