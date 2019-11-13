from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.utils import shuffle
import numpy as np

commands = open("loadData.py").read()
exec(commands)

model_path    = '../models/regr.sav'
indices       = data.index[data['testid'].isin(range(1,12))].tolist()
training_data = shuffle(np.array(data)[indices, :])
X = training_data[:, feature_columns]
Y = training_data[:, target_columns]

model =   LinearRegression()
model.fit(X, Y)
joblib.dump(model, model_path)