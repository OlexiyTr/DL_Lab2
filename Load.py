from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds


def _normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def load() -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    (ds_train, ds_test), ds_info = tfds.load(
        name='mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)

    ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)

    return ds_train, ds_test
