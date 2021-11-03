commands = open("initialize.py").read()
exec(commands)
from pandas.core.common import flatten

out_center = [scaler.data_min_[np.r_[feat]] for feat in time_series_feature_columns]
out_center.append([scaler.data_min_[np.r_[feat]] for feat in target_columns])


out_scale = [scaler.data_range_[np.r_[feat]] for feat in time_series_feature_columns]
out_scale.append([scaler.data_range_[np.r_[feat]] for feat in target_columns])

file_output = open('surro_conf.txt', 'w')
print('C nfeatures ntargets ntimesteps ..', file=file_output)
print(len(time_series_feature_columns), len(target_columns), time_series_steps, scale_min, scale_range, stress_scale, file=file_output)
print('C mins:', file=file_output)
for x in list(flatten(out_center)):
    print ("{:.60f}".format(x), file=file_output) 
print('C ranges:', file=file_output)
for x in list(flatten(out_scale)):
    print ("{:.60f}".format(x), file=file_output) 

file_output.close()
