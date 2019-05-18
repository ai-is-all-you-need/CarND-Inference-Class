import os
import numpy as np
import tensorflow as tf

from tf_classifier import TLClassifier

PATH_TO_SAVED_MODEL = os.path.join("/capstone_models/mobilenetv1", "saved_model")
assert os.path.isdir(PATH_TO_SAVED_MODEL)

IMAGE_BATCH = 5
# Image noisey
img = np.random.rand(IMAGE_BATCH, 100, 100,3) # Batch, size, channels

classifier = TLClassifier(PATH_TO_SAVED_MODEL)
example_result = classifier.get_classification(img, threshold=0.01)
print example_result

