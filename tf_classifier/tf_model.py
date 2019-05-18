import os
import numpy as np
import tensorflow as tf

class TLClassifier(object):
    def __init__(self, saved_model_path):
        assert os.path.isdir(saved_model_path), "{} is not a directory".format(saved_model_path)
        self.predictor = tf.contrib.predictor.from_saved_model(saved_model_path)


    def get_classification(self, image, threshold=0.6):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        assert len(image.shape) == 4, "image should be rank 4, with batch dimmesion on the first dimension"
        raw_predictions = self.predictor({"inputs": image})
        predictions = self.parse_predictions(raw_predictions, threshold)

        # TODO: Add logic to decide which traffic light are we seeing based on predictions

        return predictions

    @staticmethod
    def parse_predictions(predictions, threshold=0.6):
        assert type(predictions) == dict, "Prediction type is {}, not dict".format(type(predictions))
        frames = predictions['num_detections'].shape[0]

        evidence = []
        for frame_index in xrange(frames):
            bool_index = np.greater_equal(predictions['detection_scores'][frame_index], threshold)

            detection_classes = predictions['detection_classes'][frame_index][bool_index]
            detection_scores = predictions['detection_scores'][frame_index][bool_index]
            evidence_frame = zip(detection_classes, detection_scores)

            evidence.append(evidence_frame)

        return evidence