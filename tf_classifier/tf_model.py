import os
import tensorflow as tf

class TLClassifier(object):
    def __init__(self, saved_model_path):
        assert os.path.isdir(saved_model_path), "{} is not a directory".format(saved_model_path)
        self.predictor = tf.contrib.predictor.from_saved_model(saved_model_path)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        assert len(image.shape) == 4, "image should be rank 4, with batch dimmesion on the first dimmension"
        predictions = self.predictor({"inputs": image})

    @staticmethod
    def parse_predictions(predictions):
        assert type(predictions) == dict, "Prediction type is {}, not dict".format(type(predictions))
        # TODO: Finish up
        raise NotImplementedError
