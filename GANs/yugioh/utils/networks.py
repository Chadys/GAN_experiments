import tensorflow as tf


# TODO change to work with NCHW images

def _generator_helper(noise, weight_decay, is_training):
    """Core generator.

    Args:
        noise: A 2D Tensor of shape [batch size, noise dim].
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """

    # Here is the correct formula for computing the size of the output with tf.layers.conv2d_transpose():
    #
    # # Padding==Same:
    # H = H1 * stride
    #
    # # Padding==Valid
    # H = (H1 - 1) * stride + HF
    #
    # where, H = output size, H1 = input size, HF = height of filter

    net = tf.reshape(noise, [-1, 1, 1, noise.shape[1]])
    net = tf.layers.batch_normalization(tf.layers.conv2d_transpose(net, 256, [8, 8], strides=1, padding='valid', activation=tf.nn.relu), training=is_training)
    net = tf.layers.batch_normalization(tf.layers.conv2d_transpose(net, 128, [4, 4], strides=2, padding='same', activation=tf.nn.relu), training=is_training)
    net = tf.layers.batch_normalization(tf.layers.conv2d_transpose(net, 64, [4, 4], strides=2, padding='same', activation=tf.nn.relu), training=is_training)
    net = tf.layers.batch_normalization(tf.layers.conv2d_transpose(net, 32, [4, 4], strides=2, padding='same', activation=tf.nn.relu), training=is_training)
    # Make sure that generator output is in the same range as `inputs`
    # ie [-1, 1].
    net = tf.layers.conv2d_transpose(net, 3, [4, 4], strides=2, padding='same', activation=tf.tanh)
    return net


def generator(noise, weight_decay=2.5e-5, is_training=True):
    """Generator to produce unconditional images.

    Args:
        noise: A single Tensor representing noise.
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
    return _generator_helper(noise, weight_decay, is_training)


def _leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def _discriminator_helper(img, weight_decay):
    """Core discriminator.

    Args:
        img: Real or generated images. Should be in the range [-1, 1].
        weight_decay: The L2 weight decay.

    Returns:
        Final fully connected discriminator layer. [batch_size, 2048].
    """

    net = tf.layers.conv2d(img, 32, [4, 4], strides=2, padding='same', activation=_leaky_relu)
    net = tf.layers.batch_normalization(tf.layers.conv2d(net, 64, [4, 4], strides=2, padding='same', activation=_leaky_relu), training=True)
    net = tf.layers.batch_normalization(tf.layers.conv2d(net, 128, [4, 4], strides=2, padding='same', activation=_leaky_relu), training=True)
    net = tf.layers.batch_normalization(tf.layers.conv2d(net, 256, [8, 8], strides=2, padding='same', activation=_leaky_relu), training=True)
    # net = tf.layers.batch_normalization(tf.layers.conv2d(net, 512, [4, 4], strides=2, padding='same', activation=_leaky_relu), training=True)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)

    return net


def discriminator(img, weight_decay=2.5e-5):
    """Discriminator network on unconditional images.

    Args:
        img: Real or generated images. Should be in the range [-1, 1].
        weight_decay: The L2 weight decay.

    Returns:
        Logits for the probability that the image is real.
    """
    # img = img + tf.random_normal(shape=tf.shape(img),
    #                              mean=0.0,
    #                              stddev=tf.divide(0.5, tf.to_float(tf.train.get_global_step()+1)),
    #                              dtype=tf.float32)
    # img = img + tf.random_normal(shape=tf.shape(img), mean=0.0, stddev=0.1, dtype=tf.float32)
    return _discriminator_helper(img, weight_decay)
