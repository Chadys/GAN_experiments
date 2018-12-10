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
    with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.fully_connected, tf.contrib.layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.batch_norm], is_training=is_training):
            net = tf.contrib.layers.fully_connected(noise, 2048)
            net = tf.contrib.layers.fully_connected(net, 16 * 16 * 128)
            net = tf.reshape(net, [-1, 16, 16, 128])
            net = tf.contrib.layers.conv2d_transpose(net, 64, [4, 4], stride=2)
            net = tf.contrib.layers.conv2d_transpose(net, 32, [4, 4], stride=2)
            net = tf.contrib.layers.conv2d_transpose(net, 16, [4, 4], stride=2)
            # Make sure that generator output is in the same range as `inputs`
            # ie [-1, 1].
            net = tf.contrib.layers.conv2d(net, 3, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

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
    return tf.nn.leaky_relu(x, alpha=0.01)


def _discriminator_helper(img, weight_decay):
    """Core discriminator.

    Args:
        img: Real or generated images. Should be in the range [-1, 1].
        weight_decay: The L2 weight decay.

    Returns:
        Final fully connected discriminator layer. [batch_size, 2048].
    """

    with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected],
            activation_fn=_leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm,
            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            biases_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
        net = tf.contrib.layers.conv2d(img, 32, [4, 4], stride=2)
        net = tf.contrib.layers.conv2d(net, 64, [4, 4], stride=2)
        net = tf.contrib.layers.conv2d(net, 128, [4, 4], stride=2)
        net = tf.contrib.layers.flatten(net)
        net = tf.contrib.layers.fully_connected(net, 2048, normalizer_fn=tf.contrib.layers.layer_norm)

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
    net = _discriminator_helper(img, weight_decay)
    return tf.contrib.layers.linear(net, 1)
