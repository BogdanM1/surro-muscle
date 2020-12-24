import numpy as np 
import tensorflow as tf
from tensorflow.python.platform import gfile
from nested_lstm import NestedLSTM

sample = np.array( [[
                    [0, 1, 0, 0],
                    [ 0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [ 0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [ 0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [ 0, 1, 0, 0],
                    [0, 1, 0, 0]
                    ]] )

with tf.compat.v1.Session() as sess:	    
	with gfile.FastGFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.compat.v1.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.compat.v1.import_graph_def(graph_def)	        
		tensor_input = sess.graph.get_tensor_by_name('import/input_layer:0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/output_layer/strided_slice_3:0')	        
		predictions = sess.run(tensor_output, {tensor_input:sample})	        
		print(predictions)	        


from keras.models import load_model
commands = open("initialize.py").read()
exec(commands)
 
model = load_model('../models/model.h5', 
	  custom_objects={'NestedLSTM':NestedLSTM,'DiffGrad':DiffGrad,
	  'huber':huber_loss()})
print(model.predict(sample))
