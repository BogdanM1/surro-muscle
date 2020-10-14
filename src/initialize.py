import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, LeakyReLU
from keras import optimizers
from keras import backend as K
import itertools
import keras.initializers
from diffgrad import DiffGrad
import tensorflow_addons as tfa
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

K.set_floatx('float32')

feature_columns = [1, 2, 3, 4, 5, 6]
target_columns  = [7, 8]

time_series_steps = 11
time_series_feature_columns = np.array([2, 4, 5, 6]) 

def InputToTimeSeries(data, converged = None):
  data_count = len(data)
  nfeatures = len(data[0])
  outdata   = np.empty([data_count, time_series_steps, nfeatures])

  for i in range(0, time_series_steps):
    outdata[0, i, :] = data[0, :]

  for i in range(1, data_count):
    if(converged is None or converged[i-1]):
      for j in range(0, time_series_steps - 1):
        outdata[i, j, :] = outdata[i-1, j+1, :]
    else:
      for j in range(0, time_series_steps - 1):
        outdata[i, j, :] = outdata[i-1, j, :]
    outdata[i, time_series_steps-1, :] = data[i, :]
  return (outdata)

data = pd.read_csv("../data/dataMexie.csv")
data_noiter = pd.read_csv("../data/dataMexieNoIter.csv")
'''
data_large  = pd.read_csv("../data/dataMexieLargeModel.csv")
data_noiter_large = pd.read_csv("../data/dataMexieLargeModelNoIter.csv")

data = data.append(data_large)
data_noiter = data_noiter.append(data_noiter_large) 
del data_large
del data_noiter_large
'''

scaler = MinMaxScaler(feature_range=(0,1))

chunk_size=10000
for i in itertools.chain(np.setdiff1d(range(1,45),range(4,45,4))): 
    indices = data['testid'].isin([i])
    scaler.partial_fit(data[indices])

for start in range(0, data.shape[0], chunk_size):
  df_subset = data.iloc[start:start + chunk_size]
  if(start==0):
      data_scaled = scaler.transform(df_subset)
  else:
      data_scaled = np.append(data_scaled, scaler.transform(df_subset), axis=0)

for start in range(0, data_noiter.shape[0], chunk_size):
  df_subset = data_noiter.iloc[start:start + chunk_size]  
  if(start==0):
      data_scaled_noiter = scaler.transform(df_subset)
  else:
      data_scaled_noiter = np.append(data_scaled_noiter, scaler.transform(df_subset), axis=0)  

# huber 
def huber_loss(tolerance=.01):
    def huber(y,y_pred):
        error = y - y_pred
        is_small_error = tf.abs(error) < tolerance
        squared_loss = tf.square(error) / 2 
        linear_loss = tolerance*tf.abs(error) - tolerance*tolerance*0.5 
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber

# smape
def smape(y_true, y_pred): 
    epsilon = .1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_true - y_pred) / summ
    return smape

# define optimizer and loss 
optimizer=DiffGrad(lr=1e-6)
loss=huber_loss()
