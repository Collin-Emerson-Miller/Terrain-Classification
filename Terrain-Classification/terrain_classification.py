# import the necessary modules
from __future__ import division

import argparse

import cv2
import numpy as np
import yaml
import os
from keras.models import model_from_json
from utils import get_depth, get_video, prepare_images, class_label_overlay, get_spaced_colors

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model_name", help="Specify the name of the model to train.", default="squeezenet")

args = parser.parse_args()

MODEL_NAME = args.model_name.lower()

print(MODEL_NAME)

n_classes = 2

if __name__ == "__main__":

    # Load config
    with open('models/{}_config.yml'.format(MODEL_NAME), 'r') as config_file:
        config = yaml.load(config_file)

    print(config)

    n_slices = config['input_config']['n_slices']
    ratio = config['input_config']['ratio']
    image_size = config['input_config']['image_size']
    slice_height = config['input_config']['slice_height']
    slice_width = config['input_config']['slice_width']
    classes = config['input_config']['classes']
    n_slices_v = config['input_config']['n_slices_v']
    n_slices_h = config['input_config']['n_slices_h']

    # load json and create model
    json_file = open(os.path.join('models', MODEL_NAME + ".json"), 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights(os.path.join('models', MODEL_NAME + ".h5"))
    print("Loaded model from disk")

    colors = get_spaced_colors(n_classes)

    while True:

        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()

        X = prepare_images(frame, image_size, ratio, n_slices)
        X = X.astype(np.float32)

        preds = model.predict(X)
        classes = np.argmax(preds, axis=1).reshape((n_slices_h, n_slices_v))

        # Initialize the class assignment mask.
        assignment_mask = np.zeros((classes.shape[0], classes.shape[1], 3))

        # Fill class labels into assignment mask.
        for label in xrange(n_classes):
            assignment_mask[np.isin(classes, label)] = colors[label]

        assignment_mask = cv2.resize(assignment_mask, image_size, interpolation=cv2.INTER_CUBIC)

        overlay = class_label_overlay(frame, assignment_mask)

        cv2.imshow("img", frame)
        cv2.imshow("depth", depth.astype(np.uint8))
        cv2.imshow("overlay", overlay)
        print(classes)

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
