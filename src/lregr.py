from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.utils import shuffle
import numpy as np

commands = open("load_data.py").read()
exec(commands)

model_path    = '../models/regr.sav'
indices       = data.index[data['testid'].isin([1, 2, 3, 4, 5, 6])].tolist()
training_data = shuffle(np.array(data)[indices, :])
X = training_data[:, feature_columns]
Y = training_data[:, target_columns]

model =   LinearRegression()
model.fit(X, Y)
joblib.dump(model, model_path)
