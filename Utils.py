from typing import List

import tensorflow as tf
from keras import Sequential
from keras import layers


def augment(image_size: int) -> layers.Layer:
    list_layers = [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ]

    data_augmentation = Sequential(
        list_layers,
        name="data_augmentation",
    )
    return data_augmentation


def mlp(x: tf.Tensor, hidden_units: List[int], dropout_rate: float) -> tf.Tensor:
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
