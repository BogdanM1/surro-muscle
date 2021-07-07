import shutil
log_dir = "../logs"
try:
    shutil.rmtree(log_dir)
except OSError as e:
    print("Error: %s : %s" % (log_dir, e.strerror))
    
writer_tf = tf.summary.create_file_writer(log_dir)
gradslog = open(log_dir+'/gradients.csv','w')
gradslog.write('epoch,global_norm\n') 

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
  def _log_gradients(self, epoch):
    with writer_tf.as_default(), tf.GradientTape() as g:
      # here we use data to calculate the gradients
      _x_batch = tf.convert_to_tensor(X[:1000])
      _y_batch = tf.convert_to_tensor(Y[:1000])

      g.watch(_x_batch)
      _y_pred = self.model(_x_batch)  # forward-propagation
      loss = self.model.loss(y_true=_y_batch, y_pred=_y_pred)  # calculate loss
      gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation
      # In eager mode, grads does not have name, so we get names from model.trainable_weights
      for weights,grads in zip(self.model.trainable_weights,gradients):
        if(grads != None):
          tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads, step=epoch)
      norms = tf.linalg.global_norm([grads for grads in gradients if grads!=None])
      #tf.summary.scalar('global gradient norm', norms, step=epoch)     
      gradslog.write(str(epoch)+ ',' + str(norms.numpy()) +'\n') 
      tf.summary.flush()
      
  def on_epoch_end(self, epoch, logs=None):  
    # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
    # but we do need to run the original on_epoch_end, so here we use the super function. 
    super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)
    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_gradients(epoch)  