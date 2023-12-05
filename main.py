import keras
from keras.optimizers.legacy import Adam

from Draw import draw
from Load import load
from Transformer import transformer

num_classes = 10
input_shape = [28, 28, 1]
learning_rate = 1e-4
num_epochs = 10
image_size = 28
patch_size = 7
projection_dim = 256
dropout = 0.2
num_heads = 8
transformer_units = [512, 256]
n_transformer_layers = 3
mlp_head_units = [256]
num_patches = 16


def train(model, ds_train, ds_test) -> keras.callbacks.History:
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    history = model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test
    )

    _, accuracy = model.evaluate(ds_test)
    print(f"Test acc: {round(accuracy * 100, 2)}%")
    return history


def main():
    dataset_train, dataset_test = load()

    transformer_model = transformer(input_shape=input_shape,
                                    num_classes=num_classes,
                                    image_size=image_size,
                                    patch_size=patch_size,
                                    num_patches=num_patches,
                                    projection_dim=projection_dim,
                                    dropout=dropout,
                                    n_transformer_layers=n_transformer_layers,
                                    num_heads=num_heads,
                                    transformer_units=transformer_units,
                                    mlp_head_units=mlp_head_units)

    history = train(model=transformer_model, ds_train=dataset_train, ds_test=dataset_test)

    draw(history=history, filepath="figs/draw.png")
    return


if __name__ == "__main__":
    main()
