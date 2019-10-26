import pandas as pd
from sklearn.preprocessing import MinMaxScaler

feature_columns = [1, 2, 3, 4, 5, 6]
target_columns  = [7, 8]
data          = pd.read_csv("../data/dataMexie.csv")
data_noiter   = pd.read_csv("../data/dataMexieNoIter.csv")

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data)

data_scaled        = scaler.transform(data)
data_scaled_noiter = scaler.transform(data_noiter)
