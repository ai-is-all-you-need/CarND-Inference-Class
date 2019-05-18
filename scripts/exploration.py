import os
import numpy as np
import tensorflow as tf

PATH_TO_SAVED_MODEL = os.path.join("/capstone_models/mobilenetv1", "saved_model")
assert os.path.isdir(PATH_TO_SAVED_MODEL)

# Image noisey
img = np.random.rand(1, 100, 100,3) # Batch, size, channels

# Predictor
predictor = tf.contrib.predictor.from_saved_model(PATH_TO_SAVED_MODEL)
prediction_dict = predictor({"inputs": img})

print(prediction_dict)

