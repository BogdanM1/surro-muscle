from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import  Sequential

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

commands = open("load_data.py").read()
exec(commands)

model_path    = '../models/_model-new.h5'
training_data = data_scaled[data['testid'].isin(range(4,14,1))]
X = training_data[:, feature_columns]
Y = training_data[:, target_columns]

val_data = data_scaled[data['testid'].isin(range(1,4,1))]
X_val = val_data[:, feature_columns]
Y_val = val_data[:, target_columns]


model = Sequential()
model.add(Dense(120, input_dim = len(feature_columns), activation='sigmoid'))
model.add(Dropout(0.15))
model.add(Dense(80, activation='sigmoid'))
model.add(Dropout(0.15))
model.add(Dense(60, activation='sigmoid'))
model.add(Dropout(0.15))
model.add(Dense(40, activation='sigmoid'))
model.add(Dropout(0.15))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')

with open('../results/summary_mlp.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

history = model.fit(X, Y, epochs = 1000, batch_size = 64, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, save_best_only=True)])

pd.DataFrame(history.history).to_csv("../results/train.csv")
