import os
import numpy as np
import tensorflow as tf
import cv2

from tf_classifier import TLClassifier

PATH_TO_SAVED_MODEL = os.path.join("/capstone_models/mobilenetv1", "saved_model")
assert os.path.isdir(PATH_TO_SAVED_MODEL)

# IMAGE_BATCH = 1
# # Image noisey
# img = np.random.rand(IMAGE_BATCH, 100, 100,3) # Batch, size, channels

classifier = TLClassifier(PATH_TO_SAVED_MODEL)
# example_result = classifier.get_classification(img, threshold=0.5)
# print example_result

IMAGE_DATA = ["examples/test_{}.png".format(i) for i in [1, 2, 3, 4]]

for image in IMAGE_DATA:
    img = cv2.imread(image) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_array = np.array([img])
    detections =classifier.get_classification(img, threshold=1.2)

    print "image: {} has the following {}".format(image, detections) 


