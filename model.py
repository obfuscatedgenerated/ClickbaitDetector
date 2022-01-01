import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from keras import backend as K

def relu_max(x):
    return K.relu(x, max_value=1)

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
    model.add(tf.keras.layers.Dense(1, activation=relu_max))
    model.summary()
    print("Compiling...")
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
    print("Done!")
    return model