class SAM(tf.keras.layers.Layer):

  def __init__(self, kernel_size, **kwargs):
      super(SAM, self).__init__(**kwargs)

      # self.kernel_size = kernel_size

      self.concat = tf.keras.layers.Concatenate(axis=-1)
      self.conv   = tf.keras.layers.Conv2D(filters = 1, 
                                           kernel_size=Kernel_size, 
                                           padding='same',  
                                           strides = 1, 
                                           activation = 'sigmoid',  
                                           kernel_initializer='he_normal',
                                           use_bias = False)

  def call(self,inputs):
      
      x11 = tf.reduce_mean(inputs ,axis=-1, keepdims =True)
      x22 = tf.reduce_max(inputs,axis=-1,keepdims  =True)

      x   = self.concat([x11, x22])
      x   = self.conv(x)
      x   = tf.keras.layers.multiply([x, inputs])
             
      return x
