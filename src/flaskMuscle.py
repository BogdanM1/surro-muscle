from flask import Flask
from flask import request
import tensorflow as tf
from keras.models import load_model
import numpy as np
from sklearn.externals import joblib
import os
import sys

app = Flask(__name__)

commands = open("timeSeries.py").read()
exec(commands)

time_series_start = True
nqp = 4
time_series_input  = np.zeros((1, time_series_steps, len(time_series_feature_columns)))

models_directory = "../models/"
models = {}
for file_name in os.listdir(models_directory):
  model_path  = os.path.join(models_directory,file_name)
  session = tf.Session()
  graph = tf.get_default_graph()
  with graph.as_default(), session.as_default():
      model = load_model(model_path) if(file_name.endswith('.h5')) else joblib.load(model_path)
  models[file_name]  = {}
  models[file_name]['model'] = model
  models[file_name]['session'] = session
  models[file_name]['graph'] = graph

@app.route('/save_net', methods = ['POST'])
def loadNet():
  global models
  params = request.form.to_dict()
  netname = params['netname']
  model_export = request.files['network']
  model_path = os.path.join(models_directory, netname)
  model_export.save(model_path)
  graph = tf.get_default_graph()
  session = tf.Session()
  with graph.as_default(), session.as_default():
      model = load_model(model_path)
  models[netname] = {}
  models[netname]['model'] = model
  models[netname]['session'] = session
  models[netname]['graph'] = graph
  return "OK"

@app.route('/sigdsig', methods=['POST'])
def prediction():
  global models

  params     = request.form.to_dict()
  netname    = params['netname']
  model      = models[netname]['model']
  graph      = models[netname]['graph']
  session    = models[netname]['session']
    
  activation_prev = params['activation_prev'].split(',')
  activation = params['activation'].split(',')
  stretch_prev = params['stretch_prev'].split(',')
  stretch = params['stretch'].split(',')
  sigma_prev = params['sigma_prev'].split(',')
  delta_sigma_prev = params['delta_sigma_prev'].split(',')

  activation_prev = np.array(activation_prev, dtype='f')
  activation = np.array(activation, dtype='f')
  stretch_prev = np.array(stretch_prev, dtype='f')
  stretch = np.array(stretch, dtype='f')
  sigma_prev = np.array(sigma_prev, dtype='f')
  delta_sigma_prev = np.array(delta_sigma_prev, dtype='f')
  input_matrix = np.column_stack((activation_prev,activation, stretch_prev, stretch, sigma_prev, delta_sigma_prev))
  input_matrix_scaled = (input_matrix - scaler.data_min_[np.r_[feature_columns]]) / scaler.data_range_[np.r_[feature_columns]]

  if(netname.endswith('.h5')):
    with graph.as_default(), session.as_default():
      predicted = model.predict(input_matrix_scaled)
    sigma_predicted = predicted[:, 0] * scaler.data_range_[target_columns[0]] + scaler.data_min_[target_columns[0]]
    dsigma_predicted = predicted[:, 1] * scaler.data_range_[target_columns[1]] + scaler.data_min_[target_columns[1]]
  else:
    predicted = model.predict(input_matrix)
    sigma_predicted = predicted[:, 0]
    dsigma_predicted = predicted[:, 1]
  
  result = ""
  count = len(sigma_predicted)
  for i in range(count):
    result += str(sigma_predicted[i]) + "#" + str(dsigma_predicted[i]) + ","
  return result

@app.route('/start-time_series', methods=['POST'])
def start():
  global time_series_start, nqp, time_series_input
  time_series_start = True
  params     = request.form.to_dict()
  nqp        = int(params['nqp'])
  time_series_input = np.zeros((nqp, time_series_steps, len(time_series_feature_columns)))
  return "OK"

@app.route('/sigdsig-time_series', methods=['POST'])
def prediction_time_series():
  global time_series_start, models, nqp

  params     = request.form.to_dict()
  netname    = params['netname']
  model      = models[netname]['model']
  graph      = models[netname]['graph']
  session    = models[netname]['session']
  
  activation = params['activation'].split(',')
  stretch    = params['stretch'].split(',')
  sigma_prev = params['sigma_prev'].split(',')
  delta_sigma_prev = params['delta_sigma_prev'].split(',')
  converged = params['converged']
  
  activation = np.array(activation, dtype='f')
  stretch = np.array(stretch, dtype='f')
  sigma_prev = np.array(sigma_prev, dtype='f')
  delta_sigma_prev = np.array(delta_sigma_prev, dtype='f')  
  input_matrix = np.column_stack((activation, stretch, sigma_prev, delta_sigma_prev))
  input_matrix_scaled = (input_matrix - scaler.data_min_[np.r_[time_series_feature_columns]]) / scaler.data_range_[np.r_[time_series_feature_columns]]

  predicted = np.zeros((nqp, 2))
  if time_series_start:
      time_series_start = False
      for i in range(0, nqp):
        for j in range(0, time_series_steps):
          time_series_input[i, j, :] = input_matrix_scaled[i, :]
  else:
      if(int(converged) == 1):
        for i in range(0, nqp):
          for j in range(0, time_series_steps - 1):
              time_series_input[i, j, :] = time_series_input[i, j + 1, :]
      for i in range(0, nqp):
        time_series_input[i, time_series_steps - 1,] = input_matrix_scaled[i, :]

  with graph.as_default(), session.as_default():
    time_series_predicted = model.predict(time_series_input)
    if(len(time_series_predicted.shape)==2):
        predicted[:,:] = time_series_predicted[:,:]
    else:
        predicted[:, :] = time_series_predicted[:, time_series_steps-1, :]

  sigma_predicted  = predicted[:, 0] * scaler.data_range_[target_columns[0]] + scaler.data_min_[target_columns[0]]
  dsigma_predicted = predicted[:, 1] * scaler.data_range_[target_columns[1]] + scaler.data_min_[target_columns[1]]

  result = ""
  count = len(sigma_predicted)
  for i in range(count):
    result += str(sigma_predicted[i]) + "#" + str(dsigma_predicted[i]) + ","
  return result

app.run(port = 8000, host = "147.91.204.14", debug = True)
