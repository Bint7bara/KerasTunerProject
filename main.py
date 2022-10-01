import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#import matplotlib.pyplot as plt
import numpy as np

seed = 7
# load pima indians dataset
dataset = np.loadtxt("NewGencode4DLTraining.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:5]
y = dataset[:,5]

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 4)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=20,
                                            max_value=200,
                                            step=10),
                               activation='relu'))
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
    project_name='Gencode2')

tuner.search_space_summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tuner.search(X_train, y_train,
             epochs=150,
             validation_data=(X_test, y_test))

tuner.results_summary()









