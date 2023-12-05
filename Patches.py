import tensorflow as tf
from keras.layers import Layer


class Patches(Layer):

    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def __call__(self, images):
        # Виклик класу Patches зображеннями для розділення на патчі.
        batch_size = tf.shape(images)[0]

        # Видобуття патчів з зображень заданого розміру та кроку.
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Визначення розмірів патчів та їх перетворення до підходящого вигляду.
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
