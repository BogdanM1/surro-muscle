from flask import Flask, request
import tensorflow as tf
from keras.models import load_model
from sklearn.externals import joblib
import numpy as np
import os
import sys
import json
from keras_self_attention import SeqSelfAttention
from keras_radam import RAdam

app = Flask(__name__)

commands = open("timeSeries.py").read()
exec(commands)
simulation_data = {}
models = {}
models_directory = "../models/"
for file_name in os.listdir(models_directory):
  model_path  = os.path.join(models_directory,file_name)
  tf_session = tf.Session()
  graph = tf.get_default_graph()
  with graph.as_default(), tf_session.as_default():
      model = load_model(model_path, custom_objects={'huber':huber_loss(),
      'SeqSelfAttention':SeqSelfAttention,
      'RAdam':RAdam}) if(file_name.endswith('.h5')) else joblib.load(model_path)

  models[file_name]  = {}
  models[file_name]['model'] = model
  models[file_name]['session'] = tf_session
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
  tf_session = tf.Session()
  with graph.as_default(), tf_session.as_default():
      model = load_model(model_path)
  models[netname] = {}
  models[netname]['model'] = model
  models[netname]['session'] = tf_session
  models[netname]['graph'] = graph
  return "OK"

@app.route('/start', methods=['POST'])
def start():
  global simulation_data
  params = request.form.to_dict()
  simulation_id = params['simulation_id']
  simulation_data[simulation_id] = {}
  simulation_data[simulation_id]['nqp'] = params['nqp']
  simulation_data[simulation_id]['netname'] = params['netname']
  simulation_data[simulation_id]['time_series'] = params['time_series']
  npq = int(simulation_data[simulation_id]['nqp'])
  if (int(simulation_data[simulation_id]['time_series']) == 1):
    simulation_data[simulation_id]['ts_input'] = np.zeros((npq, time_series_steps, len(time_series_feature_columns)))
  simulation_data[simulation_id]['ts_start'] = True
  return "OK"

@app.route('/end', methods=['POST'])
def end():
  global simulation_data
  params = request.form.to_dict()
  simulation_id = params['simulation_id']
  del simulation_data[simulation_id]
  if(not simulation_data):
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()   
  return "OK"

@app.route('/sigdsig', methods=['POST'])
def prediction():
  global models, simulation_data

  params     = request.form.to_dict()
  simulation_id  = params['simulation_id']
  netname    = simulation_data[simulation_id]['netname']
  time_series = int(simulation_data[simulation_id]['time_series'])
  model      = models[netname]['model']
  graph      = models[netname]['graph']
  tf_session = models[netname]['session']
  b_neural_net = netname.endswith('.h5')

  activation = np.array(params['activation'].split(','), dtype='f')
  stretch =  np.array(params['stretch'].split(','), dtype='f')
  sigma_prev = np.array(params['sigma_prev'].split(','), dtype='f')
  delta_sigma_prev = np.array(params['delta_sigma_prev'].split(','), dtype='f') 

  if(time_series == 0):
    activation_prev = np.array(params['activation_prev'].split(','), dtype='f')
    stretch_prev = np.array(params['stretch_prev'].split(','), dtype='f')
    input = np.column_stack((activation_prev,activation, stretch_prev, stretch, sigma_prev, delta_sigma_prev))
    if (b_neural_net):
      input = (input - scaler.data_min_[np.r_[feature_columns]]) / scaler.data_range_[np.r_[feature_columns]]
  else:
    converged = int(params['converged'])
    nqp = int(simulation_data[simulation_id]['nqp'])
    time_series_start = bool(simulation_data[simulation_id]['ts_start'])
    input = simulation_data[simulation_id]['ts_input']
    current_input = np.column_stack((activation, stretch, sigma_prev, delta_sigma_prev))
    current_input = (current_input - scaler.data_min_[np.r_[time_series_feature_columns]]) / scaler.data_range_[np.r_[time_series_feature_columns]]
    predicted = np.zeros((nqp, 2))
    if time_series_start:
        simulation_data[simulation_id]['ts_start'] = False
        for i in range(0, nqp):
          for j in range(0, time_series_steps):
            input[i, j, :] = current_input[i, :]
    else:
        if(converged == 1):
          for i in range(0, nqp):
            for j in range(0, time_series_steps - 1):
                input[i, j, :] = input[i, j + 1, :]
        for i in range(0, nqp):
          input[i, time_series_steps - 1,] = current_input[i, :]
    simulation_data[simulation_id]['ts_input'] = input

  with graph.as_default(), tf_session.as_default():
    tmp_predicted = model.predict(input)
    if (len(tmp_predicted.shape) == 2):
      predicted = tmp_predicted
    else:
      predicted[:, :] = tmp_predicted[:, time_series_steps - 1, :]
  if (b_neural_net):
    sigma_predicted = predicted[:, 0] * scaler.data_range_[target_columns[0]] + scaler.data_min_[target_columns[0]]
    dsigma_predicted = predicted[:, 1] * scaler.data_range_[target_columns[1]] + scaler.data_min_[target_columns[1]]
  else:
    sigma_predicted = predicted[:, 0]
    dsigma_predicted = predicted[:, 1]
  
  result = ""
  count = len(sigma_predicted)
  for i in range(count):
    result += str(sigma_predicted[i]) + "#" + str(dsigma_predicted[i]) + ","
  return result

app.run(port = 8000, host = "medflow.bioirc.ac.rs", debug = False)
