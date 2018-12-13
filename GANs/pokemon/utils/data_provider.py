import os

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def parse_fn(record, shape):
    features = {
        'image_formatted': tf.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.parse_single_example(record, features)
    image = tf.image.decode_image(parsed_record['image_formatted'])

    image = tf.reshape(image, shape)  # NHWC format
    image = tf.image.resize_images(image, [FLAGS.image_dims]*2)
    # image = tf.transpose(image, [2, 0, 1]) # NCHW format
    image = (tf.to_float(image) - 128.0) / 128.0

    noise = tf.random_normal([FLAGS.noise_dims])

    return noise, image


def provide_data(dataset_dir, shape):
    """Provides batches of images.

    Args:
      dataset_dir: The directory where the data can be found.
      shape: the shape of the data, needed by map
    """
    dataset = tf.data.TFRecordDataset(os.path.join(dataset_dir, "pokemon_train.tfrecord"))
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.map(map_func=lambda record: parse_fn(record, shape), num_parallel_calls=FLAGS.num_parallel_calls)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=FLAGS.batch_size, drop_remainder=True)  # drop_remainder needed because gan
    # summaries need first dim to be a defined static shape
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
    return dataset
