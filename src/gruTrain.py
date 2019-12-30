from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, GRU, CuDNNGRU, Bidirectional
from keras.models import  Sequential
from numpy.random import seed
from tensorflow import set_random_seed 
from keras_radam import RAdam
from keras_adabound import AdaBound
from keras_lookahead import Lookahead
from keras_multi_head import MultiHead
from keras_self_attention import SeqSelfAttention

seed(1)
set_random_seed(2)

commands = open("timeSeries.py").read()
exec(commands)

opt = AdaBound()
#opt = Lookahead(opt)
model_path    = '../models/model-gru-selfAttention.h5'

X = []
Y = []
for i in range(1,14,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y)
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(14,15,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:, time_series_feature_columns],np.array(data.loc[indices,'converged'])):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled[indices][:, target_columns], np.array(data.loc[indices,'converged'])):
        Y_val.append(y)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

model = Sequential()
model.add(Bidirectional(GRU(512, input_shape = (time_series_steps, len(time_series_feature_columns)), activation='sigmoid', return_sequences=True)))
model.add(Dropout(.1))
model.add(Bidirectional(GRU(256, activation='sigmoid', return_sequences=True)))
model.add(Dropout(.1))
model.add(Bidirectional(GRU(64, activation ='sigmoid', return_sequences=True)))
model.add(SeqSelfAttention())
model.add(Dropout(.1))
model.add(Dense(2))
model.compile(loss=huber_loss(), optimizer=opt)

history = model.fit(X, Y, epochs = 1000, batch_size = 512, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])
pd.DataFrame(history.history).to_csv("../results/train-gru.csv")

