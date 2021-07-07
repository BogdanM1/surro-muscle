from sklearn.linear_model import LinearRegression, HuberRegressor
import joblib
from sklearn.utils import shuffle
import numpy as np

commands = open("initialize.py").read()
exec(commands)

model_path    = '../models/regr.sav'
indices       = data.index[data['testid'].isin(range(1,95,2))].tolist()
training_data = shuffle(np.array(data)[indices, :])
X = training_data[:, feature_columns]
Y = training_data[:, target_columns]

model =   LinearRegression()
model.fit(X, Y)
joblib.dump(model, model_path)

model_path    = '../models/huberSigma.sav'
modelSig = HuberRegressor(max_iter=1000, tol=1e-7, alpha=1e-5)
modelSig.fit(X, Y[:,0])
joblib.dump(modelSig, model_path)

model_path    = '../models/huberDSig.sav'
modelDSig = HuberRegressor(max_iter=1000,tol=1e-7, alpha=1e-5)
modelDSig.fit(X, Y[:,1])
joblib.dump(modelDSig, model_path)

'''
print('linear regression coefficients:')
print(' %.20e' % model.intercept_[0], end='')
for coef in model.coef_[0]:
  print(' %.20e' % coef, end='')
print('%.20e' % model.intercept_[1], end='')
for coef in model.coef_[1]:
  print(' %.20e' % coef, end='')  
print('')
'''
#print('huber regression coefficients:')
print(' %.20e' % modelSig.intercept_, end='')
for coef in modelSig.coef_:
  print(' %.20e' % coef, end='')
print(' %.20e' % modelDSig.intercept_, end='')
for coef in modelDSig.coef_:
  print(' %.20e' % coef, end='')
print('')
