# import numpy as np
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.reset_default_graph()
with tf.Session(config=config):
    sess = tf.get_default_session()
    # tf.reset_default_graph()
    sess.as_default()
    graph = tf.Graph()
    graph.as_default()
