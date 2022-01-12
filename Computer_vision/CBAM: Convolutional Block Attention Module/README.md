## **CBAM** is a light weight attention module which was designed to be integrated as supplementary block to Convnets to enhance its representational power without increasing its complexity.

**It contains two main modules:**

1. Channel Attention Module (CAM).
2. Spatial Attention Module (SAM).

# CAM Equation:

# CAM code:

```
class cam(tf.keras.layers.Layer):
    def __init__(self, ratio, **kwargs):


        super(cam, self).__init__(**kwargs)

        self.ratio = ratio

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.maxpool = tf.keras.layers.GlobalMaxPooling2D()

        
        self.add = tf.keras.layers.Add()
        self.act1 = tf.keras.layers.Activation('relu')
        self.act2 = tf.keras.layers.Activation('sigmoid')

    def build(self, input_shape):
        self.w1 = self.add_weight(shape = (input_shape[-1], input_shape[-1]//self.ratio), 
                                initializer = tf.keras.initializers.he_normal(),  
                                trainable = True, name='weights_1')  

        self.b1 = self.add_weight(shape = (input_shape[-1]//self.ratio,), 
                                initializer = tf.keras.initializers.zeros(),
                                trainable = True, name='bias_1')

        self.w2 = self.add_weight(shape = (input_shape[-1]//self.ratio, input_shape[-1]), 
                                initializer = tf.keras.initializers.he_normal(),  
                                trainable = True, name='weights_2')  

        self.b2 = self.add_weight(shape = (input_shape[-1],), 
                                initializer = tf.keras.initializers.he_normal(),
                                trainable = True, name = 'bias_2')  
      

    def call(self, inputs):
        x1 = self.avgpool(inputs)    
        x2 = self.maxpool(inputs)

        x1 = self.act1(tf.matmul(x1, self.w1) + self.b1)
        x1 = tf.matmul(x1, self.w2) + self.b2

        x2 = self.act1(tf.matmul(x2, self.w1) + self.b1)
        x2 = tf.matmul(x2, self.w2) + self.b2

        x = self.add([x1, x2])
        x = self.act2(x)

        x = tf.keras.layers.multiply([x, inputs])

        return x

    def get_config(self):
        config = super(cam, self).get_config()
        config.update({
        "ratio": self.ratio})
        return config              
```

# *Important*
 I faced a problem when i tried to create the weights without using `name` attribute at the `self.add_weight()`, as if keras or tensorflow is saving the weights with special name, and if you tried to run your model using these custom layers, you will face this error while saving `ValueError: Unable to create group (Name already exists)`. So the solution after searching was to name the weights and it worked for me, you can read about the solution here: https://issueexplorer.com/issue/keras-team/keras-io/720
 
 




