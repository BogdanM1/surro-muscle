from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, GRU, Bidirectional, GaussianNoise, Input
from keras.layers.normalization import BatchNormalization
from keras.models import  Sequential
from numpy.random import seed
from tensorflow import set_random_seed 
from keras_self_attention import SeqSelfAttention

seed(1)
set_random_seed(2)

commands = open("timeSeries.py").read()
exec(commands)

model_path    = '../models/model-gru.h5'

X = []
Y = []
for i in range(20,29,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y)
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(16,20,1):
    indices = data_noiter['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled_noiter[indices][:, time_series_feature_columns]): 
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled_noiter[indices][:, target_columns]): 
        Y_val.append(y)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

model = Sequential()
#model.add(GaussianNoise(.1, input_shape=(time_series_steps, len(time_series_feature_columns))))
model.add(Bidirectional(GRU(128, return_sequences=True, input_shape=(time_series_steps, len(time_series_feature_columns)))))
#model.add(BatchNormalization())
model.add(Activation('gelu'))
model.add(Dropout(.1))

model.add(Bidirectional(GRU(64, return_sequences=True)))
#model.add(BatchNormalization())
model.add(Activation('gelu'))
model.add(Dropout(.1))

model.add(SeqSelfAttention())
model.add(Dense(2))
model.compile(loss='huber_loss', optimizer='radam')

history = model.fit(X, Y, epochs = 10000, batch_size = 512, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])
pd.DataFrame(history.history).to_csv("../results/train-gru.csv")

