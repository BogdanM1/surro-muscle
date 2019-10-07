from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import  Sequential
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

commands = open("load_data.py").read()
exec(commands)

model_path    = '../models/_model-new.h5'
training_data = data_scaled[data['testid'].isin([1, 2, 3, 4, 5, 6])]
X = training_data[:, feature_columns]
Y = training_data[:, target_columns]

val_data = data_scaled[data['testid'].isin([8])]
X_val = val_data[:, feature_columns]
Y_val = val_data[:, target_columns]

model = Sequential()
model.add(Dense(50, input_dim = len(feature_columns) , activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(48, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(28, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')

history = model.fit(X, Y, epochs = 500, batch_size = 32, validation_data = (X_val, Y_val), 
                    callbacks=[ModelCheckpoint(model_path, save_best_only=True)])

plt.plot(history.history['loss'])
plt.savefig('../results/train.png')
plt.close()

