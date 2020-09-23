import numpy as np 
import tensorflow as tf
from tensorflow.python.platform import gfile

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

with tf.Session() as sess:	    
	with gfile.FastGFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.import_graph_def(graph_def)	        
		tensor_input = sess.graph.get_tensor_by_name('import/input_layer:0')	        
		tensor_output = sess.graph.get_tensor_by_name('import/output_layer/BiasAdd:0')	        
		predictions = sess.run(tensor_output, {tensor_input:sample})	        
		print(predictions)	        


from keras.models import load_model
from keras_self_attention import SeqSelfAttention
from keras_radam import RAdam
from keras_layer_normalization import LayerNormalization
commands = open("initialize.py").read()
exec(commands)
 
model = load_model('../models/model-gru-tcn.h5', 
	  custom_objects={
	  'huber':huber_loss()})
print(model.predict(sample))
