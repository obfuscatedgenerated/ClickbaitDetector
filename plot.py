import tensorflow as tf
import pandas as pd
import numpy as np

def main():
    print("Loading model...")
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
    print("Model loaded.")
    print("Plotting...")
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
    print("Done!")
if __name__ == "__main__":
    main()