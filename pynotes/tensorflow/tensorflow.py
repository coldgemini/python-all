loss_value, _ = self.sess.run([self.total_loss, self.train_op],
                              feed_dict={self.in_data: data_np, self.in_label: label_np})


var = tf.global_variables()
var_to_restore = [val  for val in var if 'conv1' in val.name or 'conv2'in val.name]
saver = tf.train.Saver(var_to_restore )
saver.restore(sess, os.path.join(model_dir, model_name))
var_to_init = [val  for val in var if 'conv1' not in val.name or 'conv2'not in val.name]
tf.initialize_variables(var_to_init)

exclude = ['layer1', 'layer2']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, os.path.join(model_dir, model_name))