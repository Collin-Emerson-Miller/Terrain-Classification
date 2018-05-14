# import the necessary modules
from __future__ import division

import cv2
import numpy as np
import os
from keras.models import model_from_json
from utils import prepare_images, get_spaced_colors
import tensorflow as tf
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

graph = tf.get_default_graph()


class image_converter:

    def __init__(self, model_name, weight_path):
        self.image_pub = rospy.Publisher("/camera/rgb/image_classification", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)

        self.model_name = model_name
        self.weight_path = weight_path

        self.model_name = model_name
        self.weight_path = weight_path

        self.n_slices = 6
        self.ratio = (4, 3)
        self.image_size = (640, 480)

        self.height = self.n_slices * self.ratio[1]
        self.width = self.n_slices * self.ratio[0]

        self.colors = get_spaced_colors(self.n_classes)

        # load json and create model
        json_file = open(os.path.join(self.weight_path, self.model_name + ".json"), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(os.path.join(self.weight_path, self.model_name + ".h5"))
        print("Loaded model from disk")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        with graph.as_default():

            X = prepare_images(cv_image, self.image_size, self.ratio, self.n_slices)
            print(X.shape)
            X = X.astype(np.float32)

            preds = self.model.predict(X)
            labels = np.argmax(preds, axis=1).reshape((self.n_slices, self.n_slices))

            # Initialize the class assignment mask.
            assignment_mask = np.zeros((labels.shape[0], labels.shape[1], 3))

            # Fill class labels into assignment mask.
            for label in xrange(len(2)):
                assignment_mask[np.isin(labels, label)] = self.colors[label]

            # Expand Classifications.
            assignment_mask = cv2.resize(assignment_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Publish classifications.
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(assignment_mask, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == "__main__":

    print(os.getcwd())
    if len(sys.argv) < 3:
        model_name = "small"
        weight_path = "weights"
    else:
        weight_path = sys.argv[1]
        model_name = sys.argv[2]

    ic = image_converter(model_name, weight_path)
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
