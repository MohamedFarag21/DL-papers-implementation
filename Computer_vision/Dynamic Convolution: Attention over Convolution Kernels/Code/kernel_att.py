class kernel_att(tf.keras.layers.Layer):
    def __init__(self, ratio, no_kernels, **kwargs):
        super(kernel_att, self).__init__(**kwargs)

        self.ratio = ratio
        self.no_kernels = no_kernels
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(self.ratio)
        self.act1 = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(self.no_kernels)
        self.act2 = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        # Squeeze-excitation module code
        x = self.avgpool(inputs)
        x = self.dense1(x)  
        x = self.act1(x)
        x = self.dense2(x)
        attention_values = self.act2(x)

        return attention_values

    def config(self):

        config = super(kernel_att, self).get_config()
        config.update({"ratio":self.ratio,  
                       "no_kernel":self.no_kernels
                       })
        return config 
