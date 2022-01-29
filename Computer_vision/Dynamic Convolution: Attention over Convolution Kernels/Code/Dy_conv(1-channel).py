class DY_conv(tf.keras.layers.Layer):
    """
    An implementation of dynamic convolution layer.
    """

    def __init__(self, ratio, no_kernels, filter_size, **kwargs):
        super(DY_conv, self).__init__(**kwargs)

        self.ratio = ratio
        self.no_kernels = no_kernels
        self.filter_size = filter_size

        self.dense1 = tf.keras.layers.Dense(self.ratio)
        self.dense2 = tf.keras.layers.Dense(self.no_kernels)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.act1 = tf.keras.layers.Activation('relu')
        self.act2 = tf.keras.layers.Activation('softmax')
        self.conv_kernels = tf.keras.layers.Conv2D(self.no_kernels, (self.filter_size, self.filter_size))
        self.final_conv = tf.keras.layers.Conv2D(1, (self.filter_size, self.filter_size))

        self.bn = tf.keras.layers.BatchNormalization()

    

    def call(self, inputs):
        # Squeeze-excitation module code
        x1 = self.avgpool(inputs)
        x1 = self.dense1(x1)  
        x1 = self.act1(x1)
        x1 = self.dense2(x1)
        attention_values = self.act2(x1)
        
        
        # test variable is used to initialize the weights of the convoluitonal kernels.
        test = self.conv_kernels(inputs)

        # 1- Get the kernel weights and biases
        # We get two nested lists on for weights and one for biases
        kernel_weights = self.conv_kernels.get_weights()

        # convert to numpy array
        weights = np.array(kernel_weights[0])
        weights = np.squeeze(weights, axis = 2)

        biases = np.array(kernel_weights[1])

        # place holder for attention weights and biases
        weight_att = np.ones((weights.shape[0], weights.shape[1], weights.shape[2]))
        bias_att   = np.ones((biases.shape[0],))

        # 2- loop over kernel index and multiply each weight and bias by the attention value

        for i in range(self.no_kernels):
            # weights[:,:,i] = weights[:,:,i] * attention_values[i]
            # biases[i] = biases[i] * attention_values[i]


            weight_att[:,:,i] = attention_values[:,i]
            bias_att[i] = attention_values[:,i]

        # 3- Multiply attention values by the kernel weights

        weights = np.multiply(weights, weight_att)
        weights = np.expand_dims(weights, axis=2)

        biases  = np.multiply(biases, bias_att)

        # 4- Sum the weighted kernels & biases

        w_tilde = np.sum(weights, axis = 3)
        w_tilde = np.expand_dims(w_tilde, axis=2)

        b_tilde = np.sum(biases)
        b_tilde = np.expand_dims(b_tilde, axis=0)


        new_weights = [w_tilde, b_tilde]
        
        # test2 variable is used to initialize the weights of the final convolutional layer
        test2 = self.final_conv(inputs)
        
        self.final_conv.set_weights(new_weights)

        x = self.final_conv(inputs)
        x = self.bn(x)
        x = self.act1(x)

        return x
    
    def config(self):

        config = super(DY_conv, self).get_config()
        config.update({"ratio":self.ratio,  
                       "no_kernel":self.no_kernels,
                       "filter_size":self.filter_size,
                       })
        return config
