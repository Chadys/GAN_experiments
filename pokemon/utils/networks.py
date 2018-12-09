import tensorflow as tf

ds = tf.contrib.distributions
layers = tf.contrib.layers
tfgan = tf.contrib.gan


# TODO change to work with NCHW images

def _generator_helper(noise, weight_decay, is_training):
    """Core MNIST generator.

    This function is reused between the different GAN modes (unconditional,
    conditional, etc).

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
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)):
        with tf.contrib.framework.arg_scope(
                [layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(noise, 4096)
            net = layers.fully_connected(net, 16 * 16 * 128)
            net = tf.reshape(net, [-1, 16, 16, 128])
            net = layers.conv2d_transpose(net, 64, [4, 4], stride=4)
            net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
            net = layers.conv2d_transpose(net, 16, [4, 4], stride=2)
            # Make sure that generator output is in the same range as `inputs`
            # ie [-1, 1].
            net = layers.conv2d(net, 3, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

            return net


def generator(noise, weight_decay=2.5e-5, is_training=True):
    """Generator to produce unconditional MNIST images.

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
    """Core MNIST discriminator.

    This function is reused between the different GAN modes (unconditional,
    conditional, etc).

    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        weight_decay: The L2 weight decay.

    Returns:
        Final fully connected discriminator layer. [batch_size, 1024].
    """
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=_leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 32, [4, 4], stride=2)
        net = layers.conv2d(net, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=4)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 4096, normalizer_fn=layers.layer_norm)

        return net


def discriminator(img, weight_decay=2.5e-5):
    """Discriminator network on unconditional MNIST digits.

    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        weight_decay: The L2 weight decay.

    Returns:
        Logits for the probability that the image is real.
    """
    net = _discriminator_helper(img, weight_decay)
    return layers.linear(net, 1)
