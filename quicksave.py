import os

import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub

from model import *

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