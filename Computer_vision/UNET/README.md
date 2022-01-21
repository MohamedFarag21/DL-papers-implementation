## UNET is one of the most reputable and robust Deep CNN architectures for semantic segmentation.

I exploited UNET in my research during masters to segment retinal blood vessels.

### Original architecture

<code><img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png"></code>


### Details

*It consists of two main blocks `encoder path (encoder)` & `decoding path (decoder)`*

### Encoder part

```
conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = tf.keras.layers.Dropout(0.25)(conv4)
  pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = tf.keras.layers.Dropout(0.25)(conv5)
```

**Note: i modified the architecture slightly for my problem**

### Decoder part

```

```
