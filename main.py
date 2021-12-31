import os

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print(
    "GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE"
)

print("Dataset Credit: https://www.kaggle.com/amananandrai")

EPOCHS = 50

def create_model(features):
    print("Constructing hub layer...")
    print(
        "This may take some time if you haven't run this before since the model needs to download."
    )
    model_url = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
    hub_layer = hub.KerasLayer(
        model_url, input_shape=[], dtype=tf.string, trainable=True
    )
    print(hub_layer(features[:3]))
    print("Finishing...")
    model = tf.keras.Sequential()
    # model = tf.keras.Sequential([layers.Dense(64), layers.Dense(1)])
    # body = tf.keras.Sequential([layers.Dense(64), layers.Dense(1)])
    # preprocessed_inputs = preprocessing_head(inputs)
    # result = body(preprocessed_inputs)
    # model = tf.keras.Model(inputs, result)
    # model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    print("Compiling...")
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
    print("Done!")
    return model


def main():
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
    model = create_model(train_data_features)


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
    # model.load_weights(checkpoint_path)

    # Re-evaluate the model
    # loss, acc = model.evaluate(train_data_features, train_data_labels, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # Train the model
    x_val = train_data_features[:10000]
    partial_x_train = train_data_features[10000:]
    y_val = train_data_labels[:10000]
    partial_y_train = train_data_labels[10000:]
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
    main()
