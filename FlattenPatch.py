import tensorflow as tf
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Layer


class FlattenPatch(Layer):
    def __init__(self, num_patches, projection_dim):
        super(FlattenPatch, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)

        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def __call__(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
