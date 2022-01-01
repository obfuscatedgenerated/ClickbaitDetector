import tensorflow as tf
import pandas as pd
import numpy as np

def main():
    global train_data, train_data_features, train_data_labels, model
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
        results = predict([input("Input a headline: ")])
        print("Probability of being clickbait: ", results[0][0])

def imported():
    global train_data, train_data_features, train_data_labels, model
    # Load the model
    train_data = pd.read_csv(
        "./clickbait_data.csv", names=["headline", "clickbait"], skiprows=1
    )
    train_data_features = train_data.copy().headline
    train_data_labels = train_data.copy().clickbait
    train_data_features = np.array(train_data_features,dtype=object)
    train_data_labels = np.array(train_data_labels)
    model = tf.keras.models.load_model("./clickbait_model/model")
    #tf.keras.utils.plot_model(model)

def predict(value):
    return model.predict(value)

if __name__ == "__main__":
    main()
else:
    imported()