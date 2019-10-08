import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import  Sequential
from rbf_keras.rbflayer import RBFLayer, InitCentersRandom
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

commands = open("load_data.py").read()
exec(commands)

model_path    = '../models/_model-rbf.h5'
training_data = data_scaled[data['testid'].isin([1, 2, 3, 4, 5, 6])]
X = training_data[:, feature_columns]
Y = training_data[:, target_columns]

val_data = data_scaled[data['testid'].isin([7])]
X_val = val_data[:, feature_columns]
Y_val = val_data[:, target_columns]

model = Sequential()
model.add(RBFLayer(100, initializer= InitCentersRandom(X), betas=2.0,input_shape=(len(feature_columns),)))
model.add(Dense(2))
model.compile(loss='mse',optimizer='adam')

history = model.fit(X, Y, epochs = 200, batch_size = 32, validation_data = (X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, save_best_only=True)])


plt.plot(history.history['loss'])
plt.savefig('../results/train.png')
plt.close()
