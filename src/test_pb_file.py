import numpy as np 
import tensorflow as tf
from tensorflow.python.platform import gfile

sample = np.array([.1, .1, .1, .1, .1, .1])

with tf.Session() as sess:	    
	with gfile.FastGFile('../models/model.pb', 'rb') as f:	        
		graph_def = tf.GraphDef()	        
		graph_def.ParseFromString(f.read())	        
		sess.graph.as_default()	        
		g_in = tf.import_graph_def(graph_def)

tensor_output = sess.graph.get_tensor_by_name('dense_1/kernel')	    
tensor_input = sess.graph.get_tensor_by_name('dense_4/kernel')	    
predictions = sess.run(tensor_output, {tensor_input:sample})	    
print(predictions)