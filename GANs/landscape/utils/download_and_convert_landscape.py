
import os
import sys

import tensorflow as tf

_FOLDER_PATH = '../../images/landscape'

_IMAGE_SIZE = 256, 256
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
    images = []
    for data_filename in os.listdir(_FOLDER_PATH):
        if not ('.jpg' == data_filename[-4:] or '.jpeg' == data_filename[-5:]):
            continue
        with tf.gfile.GFile(os.path.join(_FOLDER_PATH, data_filename), 'rb') as f:
            images.append(f.read())
    return images


def _add_to_tfrecord(tfrecord_writer):
    """Loads data from the jpg images files and writes files to a TFRecord.

  Args:
    tfrecord_writer: The TFRecord writer to use for writing.
  """
    images = _extract_images()

    with tf.Graph().as_default():
        for index, image in enumerate(images):
            sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, len(images)))
            sys.stdout.flush()

            example = image_to_tfexample(image, 'jpg'.encode(), get_shape())
            tfrecord_writer.write(example.SerializeToString())

def _long_add_to_tfrecord(tfrecord_writer):
    """Takes soooo much time, never use, do preprocessing with opencv instead
    """
    with tf.Graph().as_default():
        with tf.Session('') as sess:
            files_list = os.listdir(_FOLDER_PATH)
            for index, data_filename in enumerate(files_list):
                sys.stdout.write('\r>> Converting image %d/%d' % (index+1, len(files_list)))
                sys.stdout.flush()
                if not ('.jpg' == data_filename[-4:] or '.jpeg' == data_filename[-5:]):
                    continue
                with tf.gfile.GFile(os.path.join(_FOLDER_PATH, data_filename), 'rb') as f:
                    image = tf.image.decode_jpeg(f.read(), channels=3)
                    image = tf.image.resize_image_with_pad(image, _IMAGE_SIZE[0], _IMAGE_SIZE[1],
                                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    image = tf.image.encode_jpeg(image)
                    image = sess.run(image)

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
    return os.path.join(dataset_dir, 'landscape_%s.tfrecord' % split_name)


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

    print('\nFinished converting the Landscape dataset!')


def get_shape():
    return _IMAGE_SIZE[0], _IMAGE_SIZE[1], _NUM_CHANNELS
