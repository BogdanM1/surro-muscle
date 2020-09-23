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
import keras.initializers
from mytcn import TCN
from nested_lstm import NestedLSTM
from numpy.random import seed
import tensorflow
import math

commands = open("initialize.py").read()
exec(commands)

_seed = 137
seed(_seed)
tf.random.set_seed(_seed)



dim_lstm_neurons = Integer(low= 16, high=128, name='lstm_neurons')
dim_tcn_neurons = Integer(low= 16, high=128, name='tcn_neurons')
dim_tcn_dilations = Categorical(categories =[ 4, 8, 16, 32], name='tcn_dilations') 
dimensions = [dim_lstm_neurons, dim_tcn_neurons, dim_tcn_dilations]
default_parameters = [128, 128, 4]

def create_model(lstm_neurons, tcn_neurons, tcn_dilations):
  lecun_normal = keras.initializers.lecun_normal(seed=_seed)
  orthogonal = keras.initializers.Orthogonal(seed=_seed)
  glorot_uniform = keras.initializers.glorot_uniform(seed=_seed)
  
  max_dilation_degree = int(math.log(tcn_dilations*2,2))
  dilations_list = [2**i for i in range(0, max_dilation_degree)]
  
  i = Input(shape=(time_series_steps, len(time_series_feature_columns)), name='input_layer')
  o = NestedLSTM(int(lstm_neurons), depth=2,return_sequences=True, kernel_initializer=orthogonal, recurrent_initializer=orthogonal, name='lstm1')(i) 
  o = TCN(nb_filters=int(tcn_neurons), kernel_size=4, dilations=dilations_list, activation='selu', kernel_initializer=lecun_normal, use_skip_connections=False, name='tcn1')(o) 
  o = Flatten()(o)
  o = Dense(2, name='output_layer')(o)
  
  model = Model(inputs = [i], outputs=[o])
  model.compile(loss=loss, optimizer=optimizer)

  return model 
  
X = []
Y = []
for i in itertools.chain(np.setdiff1d(range(1,45),range(4,45,4))): 
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
for i in itertools.chain(range(4,45,4)): 
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
def fitness(lstm_neurons, tcn_neurons, tcn_dilations):
    model = create_model(lstm_neurons, tcn_neurons, tcn_dilations)
    history = model.fit(X, Y, epochs=5, batch_size=16384, validation_data=(X_val, Y_val), verbose=2)
    fit_val = history.history['val_loss'][-1]

    print()
    print(lstm_neurons, tcn_neurons, tcn_dilations)
    print("fitness value: {0:.10}".format(fit_val))

    del model
    K.clear_session()
    tf.compat.v1.reset_default_graph()  

    return fit_val
    
K.clear_session()
tf.compat.v1.reset_default_graph()    
gp_result = gp_minimize(func=fitness, dimensions=dimensions, n_calls=50, noise=1e-10, n_jobs=-1, kappa = 1.96, 
        acq_func='EI', acq_optimizer='auto',
        verbose=True, x0=default_parameters)
print('gp optimization result')
print(gp_result)
print('gp best params:')
print(gp_result.x[0],gp_result.x[1],gp_result.x[2])


fig = plt.figure()
axes = plot_convergence(gp_result)
plt.savefig('../results/convergence.png', bbox_inches='tight',  dpi=300)
plt.close()

fig = plt.figure()
axes = plot_evaluations(gp_result)
plt.savefig('../results/evaluations-gp.png',  bbox_inches='tight', dpi=300)
plt.close()

fig = plt.figure()
axes = plot_objective(gp_result)
plt.savefig('../results/objective-gp.png', bbox_inches='tight', dpi=300)
plt.close()

