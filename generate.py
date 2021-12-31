import tensorflow as tf
import pandas as pd
import numpy as np

NORM_MIN = 0
NORM_MAX = 1
SCALAR_MIN = 0
SCALAR_MAX = 1

def normalise(value):
    return (((value - NORM_MIN)/(NORM_MAX - NORM_MIN))*(SCALAR_MAX - SCALAR_MIN)) + SCALAR_MIN

def main():
    # Load the model
    train_data = pd.read_csv(
        "./clickbait_data.csv", names=["headline", "clickbait"], skiprows=1
    )
    train_data_features = train_data.copy().headline
    train_data_labels = train_data.copy().clickbait
    train_data_features = np.array(train_data_features,dtype=object)
    train_data_labels = np.array(train_data_labels)
    model = tf.keras.models.load_model("./clickbait_model/model")
    model.evaluate(train_data_features, train_data_labels)
    #tf.keras.utils.plot_model(model)
    while True:
        results = model.predict([input("Input a headline: ")])
        print("Probability of being clickbait: ", normalise(results[0][0]))

if __name__ == "__main__":
    main()