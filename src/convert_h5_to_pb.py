import tensorflow_addons as tfa
from diffgrad import DiffGrad
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow as tf
from keras.models import load_model
from nested_lstm import NestedLSTM

commands = open("timeSeries.py").read()
exec(commands)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

model_path    = '../models/model-gru-tcn.h5'
K.set_learning_phase(0)	
model = load_model(model_path, custom_objects={
    'NestedLSTM': NestedLSTM}, compile=False)
model.compile(loss=smape, optimizer=DiffGrad(lr=1e-5))    
model.summary()  
frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.io.write_graph(frozen_graph, "../models/", "model.pb", as_text = False)
[print(n.name) for n in frozen_graph.node]
