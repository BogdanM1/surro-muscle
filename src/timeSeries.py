commands = open("loadData.py").read()
exec(commands)

import numpy as np
time_series_steps = 10
time_series_feature_columns = np.array([2, 4, 5, 6]) 

def InputToTimeSeries(data, converged = None):
  data_count = len(data)
  nfeatures = len(data[0])
  outdata   = np.empty([data_count, time_series_steps, nfeatures])

  for i in range(0, time_series_steps):
    outdata[0, i, :] = data[0, :]

  if (converged is not None):
    for i in range(1, data_count):
      if(converged[i-1]):
        for j in range(0, time_series_steps - 1):
          outdata[i, j, :] = outdata[i-1, j+1, :]
      else:
        for j in range(0, time_series_steps - 1):
          outdata[i, j, :] = outdata[i-1, j, :]
      outdata[i, time_series_steps-1, :] = data[i, :]
  else:
    for i in range(1,data_count - time_series_steps + 1):
      outdata[i, :, :]  = data[i:(i+time_series_steps), :]

  return (outdata)



