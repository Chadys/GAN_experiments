# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf

# The URLs where the MNIST data can be downloaded.
_FOLDER_PATH = '../images/pokemon'

_IMAGE_SIZE = 475
_NUM_CHANNELS = 3


def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
        values: A scalar or list of values.
    Returns:
    A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
        values: A string.
    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_formatted, format_string, shape):
    return tf.train.Example(features=tf.train.Features(feature={
        'format': bytes_feature(tf.compat.as_bytes(format_string)),
        'height': int64_feature(shape[0]),
        'width': int64_feature(shape[1]),
        'depth': int64_feature(shape[2]),
        'image_formatted': bytes_feature(tf.compat.as_bytes(image_formatted))
    }))


def _extract_images():
    """Extract the images into a numpy array.

    Args:
        filename: The path to an MNIST images file.
        num_images: The number of images in the file.

    Returns:
        A numpy array of shape [number_of_images, height, width, channels].
  """
    images = []
    for data_filename in os.listdir(_FOLDER_PATH):
        with tf.gfile.GFile(os.path.join(_FOLDER_PATH, data_filename), 'rb') as f:
            images.append(f.read())
    return images


def _add_to_tfrecord(tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

  Args:
    data_filename: The filename of the MNIST images.
    labels_filename: The filename of the MNIST labels.
    num_images: The number of images in the dataset.
    tfrecord_writer: The TFRecord writer to use for writing.
  """
    images = _extract_images()

    with tf.Graph().as_default():
        for index, image in enumerate(images):
            sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, len(images)))
            sys.stdout.flush()

            example = image_to_tfexample(image, 'jpg'.encode(), get_shape())
            tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
    return os.path.join(dataset_dir, 'pokemon_%s.tfrecord' % split_name)


def run(dataset_dir):
    """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')

    if tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        _add_to_tfrecord(tfrecord_writer)

    print('\nFinished converting the Pokemon dataset!')


def get_shape():
    return _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS
