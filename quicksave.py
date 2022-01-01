import os

import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

def create_model():
    print("Constructing hub layer...")
    print(
        "This may take some time if you haven't run this before since the model needs to download."
    )
    model_url = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
    hub_layer = hub.KerasLayer(
        model_url, input_shape=[], dtype=tf.string, trainable=True
    )
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
    print("Deprecated.")
    checkpoint_path = "clickbait_model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("Creating model...")
    model = create_model()
    print("Loading weights...")
    model.load_weights(checkpoint_path)
    print("Done!")
    print("Saving...")
    model.save("./clickbait_model/model")
    print("Saved!")

if __name__ == "__main__":
    main()