class CBAM(tf.keras.layers.Layer):
  """
  Convolutional block attention module as a layer
  """
  def __init__(self, ratio, kernel_size, **kwargs):
      super(CBAM ,self).__init__(**kwargs)

      self.ratio = ratio
      self.kernel_size = kernel_size

      self.cam = CAM(self.ratio)
      self.sam = SAM(self.kernel_size)

  def call(self, inputs):
      """
      A class method to do the computation
      """
      x = self.cam(inputs)
      x = self.sam(x)
      return x

  def get_config(self):
      """
      A class method to enable serliaization to be used at Functional and Sequential
      """
      config = super(CBAM, self).get_config()
      config.update({
      "ratio": self.ratio,  
      "kernel_size": self.kernel_size})
      return config  
