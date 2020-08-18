import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer  

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, Flatten, GRU
from keras.models import  Model, Sequential
from keras_radam import RAdam
import keras.initializers
from mytcn import TCN
from numpy.random import seed
import tensorflow
from tensorflow import set_random_seed 

_seed = 137
seed(_seed)
set_random_seed(_seed)

commands = open("timeSeries.py").read()
exec(commands)
model_path    = '../models/model-gru-tcn.h5'

dim_learning_rate = Categorical(categories=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], name='learning_rate')
dim_activation = Categorical(categories=['relu', 'sigmoid', 'selu', 'tanh'], name='activation')
dim_dropout = Categorical(categories=[.0, .1, .2, .4], name='dropout')
dim_rdropout =Categorical(categories=[.0, .1, .2, .4], name='recurrent_dropout')
dim_hloss = Categorical(categories=[.1, .01, .001, .0001], name='huber_loss_delta')
dim_batch_size = Categorical(categories=[1024, 2048, 4096, 8192], name='batch_size')

dimensions = [dim_learning_rate, dim_activation, dim_dropout, dim_rdropout, dim_hloss, dim_batch_size]
default_parameters = [1e-5, 'tanh', .2, .2, .01, 2048]

def create_model(learning_rate, activation, dropout, recurrent_dropout, huber_loss_delta):
  lecun_normal = keras.initializers.lecun_normal(seed=_seed)
  orthogonal = keras.initializers.Orthogonal(seed=_seed)
  glorot_uniform = keras.initializers.glorot_uniform(seed=_seed)
  
  i = Input(shape=(time_series_steps, len(time_series_feature_columns)), name='input_layer')
  o = GRU(128, activation=activation, return_sequences=True, kernel_initializer=orthogonal, recurrent_initializer=orthogonal, recurrent_dropout=recurrent_dropout, name='gru_layer1')(i) 
  o = Dropout(dropout)(o)
  o = TCN(nb_filters=64, kernel_size=4, dilations=[1,2,4], activation='selu', kernel_initializer=lecun_normal, use_skip_connections=False, name='tcn_layer1')(o) 
  o = Flatten()(o)
  o = Dense(2, name='output_layer') (o)
  
  model = Model(inputs = [i], outputs=[o])
  model.compile(loss=huber_loss(huber_loss_delta), optimizer=RAdam(learning_rate=learning_rate))
  return model 
  
X = []
Y = []
for i in itertools.chain(range(4,50), range(60,80)): 
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
    data = data[indices != True]
    data_scaled = data_scaled[indices != True]
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in itertools.chain(range(1,4), range(80,90)): 
    indices = data_noiter['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled_noiter[indices][:, time_series_feature_columns]):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled_noiter[indices][:, target_columns]):
        Y_val.append(y[-1])
    data_noiter = data_noiter[indices != True]
    data_scaled_noiter = data_scaled_noiter[indices != True]        
X_val = np.array(X_val)
Y_val = np.array(Y_val)

@use_named_args(dimensions=dimensions)  
def fitness(learning_rate, activation, dropout, recurrent_dropout, huber_loss_delta, batch_size):
    model = create_model(learning_rate, activation, dropout, recurrent_dropout, huber_loss_delta)
    history = model.fit(X, Y, epochs=10, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=2)
    loss = history.history['val_loss'][-1]

    print()
    print(learning_rate, activation, dropout, recurrent_dropout, huber_loss_delta, batch_size)
    print("Loss: {0:.10}".format(loss))
    print()

    del model
    K.clear_session()
    tensorflow.reset_default_graph()

    return loss 
    
K.clear_session()
tensorflow.reset_default_graph()    
gp_result = gp_minimize(func=fitness, dimensions=dimensions, n_calls=100, noise=0.01, n_jobs=-1,kappa = 5.5, x0=default_parameters)
print('gp optimization result')
print(gp_result)
print('gp best params:')
print(gp_result.x[0],gp_result.x[1],gp_result.x[2],gp_result.x[3],gp_result.x[4], gp_result.x[5])

'''
K.clear_session()
tensorflow.reset_default_graph()    
gbrt_result = gbrt_minimize(func=fitness, dimensions=dimensions, n_calls=100, n_jobs=-1, kappa = 5.5, x0=default_parameters)
print('gbrt optimization result')
print(gbrt_result)
print('gbrt best params:')
print(gbrt_result.x[0],gbrt_result.x[1],gbrt_result.x[2],gbrt_result.x[3],gbrt_result.x[4], gbrt_result.x[5])

results = [('gbrt_results', gbrt_result),('gp_results', gp_result)]
'''

fig = plt.figure()
axes = plot_convergence(gp_result)
plt.savefig('../results/convergence.png', bbox_inches='tight',  dpi=300)
plt.close()

'''
fig = plt.figure()
axes = plot_evaluations(gbrt_result)
plt.savefig("../results/evaluations-gbrt.png",  bbox_inches='tight', dpi=300)
plt.close()
'''

fig = plt.figure()
axes = plot_evaluations(gp_result)
plt.savefig('../results/evaluations-gp.png',  bbox_inches='tight', dpi=300)
plt.close()

'''
fig = plt.figure()
axes = plot_objective(gbrt_result)
plt.savefig('../results/objective-gbrt.png', bbox_inches='tight', dpi=300)
plt.close()
'''

fig = plt.figure()
axes = plot_objective(gp_result)
plt.savefig('../results/objective-gp.png', bbox_inches='tight', dpi=300)
plt.close()

'''
model=create_model(gbrt_result.x[0],gbrt_result.x[1],gbrt_result.x[2],gbrt_result.x[3],gbrt_result.x[4])
print(model.summary())
history = model.fit(X, Y, epochs = 30000, batch_size = gbrt_result.x[5], validation_data = (X_val, Y_val), verbose=2,
	            callbacks = [ModelCheckpoint(model_path, monitor = 'val_loss', save_best_only = True),
                    EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=3000)])
pd.DataFrame(history.history).to_csv("../results/train-grutcn.csv")
'''
