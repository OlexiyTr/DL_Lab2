from keras import Model

from FlattenPatch import FlattenPatch
from Patches import Patches
from Utils import *


def transformer(input_shape,
                num_classes: int,
                image_size: int,
                patch_size: int,
                num_patches: int,
                projection_dim: int,
                dropout: float,
                n_transformer_layers: int,
                num_heads: int,
                transformer_units: List[int],
                mlp_head_units: List[int]):

    inputs = layers.Input(shape=input_shape)

    augmented_data = augment(image_size=image_size)
    augmented_inputs = augmented_data(inputs)

    patches = Patches(patch_size)(augmented_inputs)
    encoded_patches = FlattenPatch(num_patches, projection_dim)(patches)

    for _ in range(n_transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    representation = layers.Flatten()(representation)

    representation = layers.Dropout(dropout)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout)

    outputs = layers.Dense(num_classes)(features)

    model = Model(inputs=inputs, outputs=outputs)

    return model
