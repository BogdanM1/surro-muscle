from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import  Sequential
from numpy.random import seed
from tensorflow import set_random_seed 

seed(1)
set_random_seed(2)

commands = open("loadData.py").read()
exec(commands)

model_path    = '../models/model-mlp.h5'
training_data = data_scaled[data['testid'].isin(range(1,6,1))]
X = training_data[:, feature_columns]
Y = training_data[:, target_columns]

val_data = data_scaled_noiter[data_noiter['testid'].isin(range(7,8,1))]
X_val = val_data[:, feature_columns]
Y_val = val_data[:, target_columns]

model = Sequential()
model.add(Dense(50, input_dim = len(feature_columns), activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(48, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')

history = model.fit(X, Y, epochs = 5, batch_size = 64, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, save_best_only=True)])
pd.DataFrame(history.history).to_csv("../results/train.csv")
