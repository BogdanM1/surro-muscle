import numpy as np
lstm_steps = 3
lstm_feature_columns = np.array([1, 2, 4, 5])
lstm_input = np.zeros((1, lstm_steps, len(lstm_feature_columns)))
start = True

def reshapeInputLSTM (data, lstm_steps):
  data_count = len(data) - lstm_steps + 1
  nfeatures = len(data[0])
  outdata   = np.empty([data_count, lstm_steps, nfeatures])
  for i in range(0,data_count):
    chunk         = data[i:(i+lstm_steps), :]
    outdata[i,:,:]  = chunk
  return (outdata)