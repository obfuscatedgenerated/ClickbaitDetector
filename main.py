import os
import sys

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

from model import *


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print(
    "GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE"
)

print("Dataset Credit: https://www.kaggle.com/amananandrai")

EPOCHS = 50

cansave = False


def main():
    global cansave, model
    # Mount the training data
    train_data = pd.read_csv(
        "./clickbait_data.csv", names=["headline", "clickbait"], skiprows=1
    )
    print(train_data.head())
    print("Shape: ", train_data.shape)
    print("Dimensions: ", train_data.ndim)
    train_data_features = train_data.copy().headline
    train_data_labels = train_data.copy().clickbait
    train_data_features = np.array(train_data_features,dtype=object)
    train_data_labels = np.array(train_data_labels)
    print("Features:", train_data_features)
    print("Labels:", train_data_labels)

    # Create the model
    print("Creating model...")
    model = create_model()


    # Checkpoint path
    checkpoint_path = "clickbait_model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    # Evaluate the model
    #loss, acc = model.evaluate(train_data_features, train_data_labels, verbose=2)
    #print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    # Loads the weights
    if os.path.isdir(checkpoint_path):
        model.load_weights(checkpoint_path)

    # Re-evaluate the model
    # loss, acc = model.evaluate(train_data_features, train_data_labels, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # Train the model
    x_val = train_data_features[:10000]
    partial_x_train = train_data_features[10000:]
    y_val = train_data_labels[:10000]
    partial_y_train = train_data_labels[10000:]
    cansave = True
    print("Training for ",str(EPOCHS)," epochs...")
    model.fit(
        partial_x_train,
        partial_y_train,
        batch_size=512,
        epochs=EPOCHS,
        callbacks=[cp_callback],
        validation_data=(x_val, y_val),
        verbose=1,
    )
    model.evaluate(x_val, y_val, verbose=2)
    print("Finished training, saving model...")
    model.save("./clickbait_model/model")
    print("Saved!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if cansave:
            print("Got KeyboardInterrupt, saving & exiting...")
            print("Saving model...")
            model.save("./clickbait_model/model")
            print("Saved!")
            print("Exiting...")
        else:
            print("Got KeyboardInterrupt, exiting...")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
