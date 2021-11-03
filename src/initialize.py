import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, LeakyReLU
from keras import optimizers
from keras import backend as K
import itertools
import keras.initializers
import tensorflow_addons as tfa
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


K.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
feature_columns = [1, 2, 3, 4, 5, 6]
target_columns  = [7, 8]

time_series_steps = 11
time_series_feature_columns = np.array([2, 4, 5, 6]) 

stress_scale = 30.0
scale_min = 0.0
scale_max = 1.0
scale_range = scale_max - scale_min
scaler = MinMaxScaler(feature_range=(scale_min,scale_max)) 
chunk_size = 10000
ntrains = 1
ntraine = 160


data = pd.read_csv("../data/dataMexie.csv")
data_noiter = pd.read_csv("../data/dataMexieNoIter.csv")

for i in itertools.chain(np.setdiff1d(range(ntrains,ntraine),range(ntrains+3,ntraine,4))): 
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
      
	  
# Input to time series
init_act = (scale_range*(0.0-scaler.data_min_[time_series_feature_columns[0]]))/scaler.data_range_[time_series_feature_columns[0]] + scale_min
init_stretch = (scale_range*(1.0-scaler.data_min_[time_series_feature_columns[1]]))/scaler.data_range_[time_series_feature_columns[1]] + scale_min
init_in_stress = (scale_range*(0.0-scaler.data_min_[time_series_feature_columns[2]]))/scaler.data_range_[time_series_feature_columns[2]] + scale_min
init_in_dstress = (scale_range*(0.0-scaler.data_min_[time_series_feature_columns[3]]))/scaler.data_range_[time_series_feature_columns[3]] + scale_min

init_out_stress = (scale_range*(0.0-scaler.data_min_[target_columns[0]]))/scaler.data_range_[target_columns[0]] + scale_min
init_out_dstress = (scale_range*(0.0-scaler.data_min_[target_columns[1]]))/scaler.data_range_[target_columns[1]] + scale_min

def InputToTimeSeries(data, converged = None):
  data_count = len(data)
  nfeatures = len(data[0])
  outdata   = np.empty([data_count, time_series_steps, nfeatures])

  if(nfeatures==4):
    for i in range(0, time_series_steps-1):
      outdata[0, i, :] = np.array([init_act, init_stretch, init_in_stress, init_in_dstress])    
  else:
    for i in range(0, time_series_steps-1):
      outdata[0, i, :] = np.array([init_out_stress, init_out_dstress])
    
  outdata[0, time_series_steps-1, :] =  data[0, :]

  for i in range(1, data_count):
    if(converged is None or converged[i-1]):
      for j in range(0, time_series_steps - 1):
        outdata[i, j, :] = outdata[i-1, j+1, :]
    else:
      for j in range(0, time_series_steps - 1):
        outdata[i, j, :] = outdata[i-1, j, :]
    outdata[i, time_series_steps-1, :] = data[i, :]
  return (outdata)

        
optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3, beta_1=0.99, beta_2=0.9999, clipnorm=1e-4)
loss = tf.keras.losses.Huber(58e-6)
#gpu2
