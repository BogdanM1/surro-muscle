import numpy as np
lstm_steps = 3
lstm_feature_columns = np.array([1, 3])
lstm_input = np.zeros((1, lstm_steps, len(lstm_feature_columns)))
start = True