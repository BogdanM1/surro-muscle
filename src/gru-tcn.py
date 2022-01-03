from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, Flatten, GRU
from keras.models import  Model, Sequential
from mytcn import TCN
from numpy.random import seed
from nested_lstm import NestedLSTM


commands = open("initialize.py").read()
exec(commands)
model_path    = '../models/model.h5'

_seed = 137
seed(_seed)
tf.random.set_seed(_seed)

X = []
Y = []
for i in itertools.chain(np.setdiff1d(range(ntrains,ntraine),range(ntrains+3,ntraine,4))):  
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
    data = data[indices != True]
    data_scaled = data_scaled[indices != True]   


X_val = []
Y_val = []
for i in itertools.chain(range(ntrains+3,ntraine,4)): 
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:, time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled[indices][:, target_columns], np.array(data.loc[indices,'converged'])):
        Y_val.append(y[-1])
    data = data[indices != True]
    data_scaled = data_scaled[indices != True]   


values_prev = [sample[-1][-2:] for sample in X]
values_prev_val = [sample[-1][-2:] for sample in X_val]

X = np.array(X)
Y = np.array(Y)         
Y -= np.array(values_prev)
Y *= stress_scale


X_val = np.array(X_val)
Y_val = np.array(Y_val)
Y_val -= np.array(values_prev_val)
Y_val *= stress_scale


lecun_normal = keras.initializers.lecun_normal(seed=_seed)
orthogonal = keras.initializers.Orthogonal(seed=_seed)
glorot_uniform = keras.initializers.glorot_uniform(seed=_seed)

i = Input(shape=(time_series_steps, len(time_series_feature_columns)), name='input_layer')
o = GRU(64, return_sequences=True, activation='selu', kernel_initializer=lecun_normal, recurrent_initializer=orthogonal, name='lstm1')(i) 
o = GRU(256, return_sequences=True, activation='selu', kernel_initializer=lecun_normal, recurrent_initializer=orthogonal, name='lstm2')(o) 
o = GRU(256, return_sequences=True, activation='selu', kernel_initializer=lecun_normal, recurrent_initializer=orthogonal, name='lstm3')(o) 
o = GRU(64, return_sequences=True, activation='selu', kernel_initializer=lecun_normal, recurrent_initializer=orthogonal, name='lstm4')(o) 
o = TCN(nb_filters=128, kernel_size=4, dilations=[1,2,4], activation='selu', kernel_initializer=lecun_normal, use_skip_connections=False, name='tcn1')(o) 
o = Flatten()(o)
o = Dense(2, name='output_layer')(o)

model = Model(inputs = [i], outputs=[o])
model.compile(loss=loss, optimizer=optimizer)
print(model.summary())



commands = open("ExtendedTensorboard.py").read()
exec(commands)
tensorboard = ExtendedTensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True,write_images=False,update_freq="epoch", profile_batch=2, embeddings_freq=0)  
modelcheckpoint = ModelCheckpoint(model_path, monitor = 'val_loss', save_best_only = True)
earlystopping = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=500)

model.fit(X, Y, epochs = 50000, batch_size = 16384, validation_data = (X_val, Y_val), verbose=2, callbacks = [tensorboard, modelcheckpoint, earlystopping])


