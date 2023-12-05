import keras
import pandas as pd


def draw(history: keras.callbacks.History, filepath: str):
    history_loss = pd.DataFrame(history.history, columns=["loss", "val_loss"])

    ax = history_loss.plot(xlabel="Epochs", ylabel="Validation loss (cross-entropy)")

    fig = ax.get_figure()

    fig.savefig(filepath)
    return
