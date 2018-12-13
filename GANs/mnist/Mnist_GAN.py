# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains a GANEstimator on MNIST data."""

import os
import multiprocessing

import numpy as np
import scipy.misc
import tensorflow as tf

from utils import data_provider
from utils import networks
from utils import download_and_convert_mnist

tfgan = tf.contrib.gan

FLAGS = tf.flags.FLAGS


def define_flags():
    tf.flags.DEFINE_integer('batch_size', 32,
                            'The number of images in each train batch.')

    tf.flags.DEFINE_integer('max_number_of_steps', 20000,
                            'The maximum number of gradient steps.')

    tf.flags.DEFINE_integer('noise_dims', 64,
                            'Dimensions of the generator noise vector')

    tf.flags.DEFINE_string('dataset_dir', './mnist_data/', 'Location of data.')

    tf.flags.DEFINE_string('eval_dir', '/tmp/mnist-estimator/',
                           'Directory where the results images are saved to.')

    tf.flags.DEFINE_string('model_dir', './mnist-model/',
                           'Directory where the checkpoints and model are saved.')

    tf.flags.DEFINE_integer('kmp_blocktime', 0,
                            'Sets the time, in milliseconds, that a thread should wait, after completing the '
                            'execution of a parallel region, before sleeping.')

    tf.flags.DEFINE_integer('kmp_settings', 1,
                            'Enables (true) or disables (false) the printing of OpenMP* run-time library environment '
                            'variables during program execution.')

    tf.flags.DEFINE_integer('num_intra_threads', 0,
                            'Specifies the number of threads to use. 0 will result in the value being set to the '
                            'number of logical cores')

    tf.flags.DEFINE_string('kmp_affinity', 'granularity=fine,verbose,compact,1,0',
                           'Enables the run-time library to bind threads to physicalprocessing units.')

    tf.flags.DEFINE_integer('num_parallel_readers', multiprocessing.cpu_count(),
                            'Level of parallelism.')

    tf.flags.DEFINE_integer('num_parallel_calls', multiprocessing.cpu_count(),
                            'Level of parallelism.')

    tf.flags.DEFINE_integer('prefetch_buffer_size', 1,
                            'Number of element in prefetch element buffer '
                            '(should be equal to number of element consumed by one training step).')

    tf.flags.DEFINE_integer('shuffle_buffer_size', 1000,
                            'The number of elements from this dataset from which the new dataset will sample.')

    os.environ["KMP_BLOCKTIME"] = str(FLAGS.kmp_blocktime)
    os.environ["KMP_SETTINGS"] = str(FLAGS.kmp_settings)
    os.environ["KMP_AFFINITY"] = FLAGS.kmp_affinity
    if FLAGS.num_intra_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(FLAGS.num_intra_threads)


def _unconditional_generator(noise, mode):
    """MNIST generator with extra argument for tf.Estimator's `mode`."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    return networks.unconditional_generator(noise, is_training=is_training)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    define_flags()

    if not tf.gfile.Exists(FLAGS.dataset_dir):
        tf.gfile.MakeDirs(FLAGS.dataset_dir)

        download_and_convert_mnist.run(FLAGS.dataset_dir)
        # tfslimprovider.run(FLAGS.dataset_dir)
    shape = download_and_convert_mnist.get_shape()

    # config = tf.ConfigProto()
    # config.gpu_options.allocator_type = 'BFC'
    # Initialize GANEstimator with options and hyperparameters.
    gan_estimator = tfgan.estimator.GANEstimator(
        generator_fn=_unconditional_generator,
        discriminator_fn=networks.unconditional_discriminator,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
        add_summaries=tfgan.estimator.SummaryType.IMAGES,
        model_dir=FLAGS.model_dir)
    # config = tf.estimator.RunConfig(session_config=config))

    # Train estimator.
    gan_estimator.train(lambda: data_provider.provide_data(FLAGS.dataset_dir, shape),
                        max_steps=FLAGS.max_number_of_steps)

    # Run inference.
    prediction_iterable = gan_estimator.predict(lambda : tf.random_normal([36, FLAGS.noise_dims]))
    predictions = [next(prediction_iterable) for _ in range(36)]

    # Nicely tile.
    image_rows = [np.concatenate(predictions[i:i + 6], axis=0) for i in
                  range(0, 36, 6)]
    tiled_image = np.concatenate(image_rows, axis=1)

    # Write to disk.
    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    scipy.misc.imsave(os.path.join(FLAGS.eval_dir, 'unconditional_gan.png'),
                      np.squeeze(tiled_image, axis=2))


if __name__ == '__main__':
    tf.app.run()
