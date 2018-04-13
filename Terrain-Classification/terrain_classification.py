# import the necessary modules
from __future__ import division

import argparse

import cv2
import numpy as np
import yaml
import os
from keras.layers import Input
from keras.models import model_from_json
from utils import get_depth, get_video, prepare_images

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model_name", help="Specify the name of the model to train.", default="squeezenet")

args = parser.parse_args()

MODEL_NAME = args.model_name.lower()

print(MODEL_NAME)

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

    while True:

        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()

        X = prepare_images(frame, image_size, ratio, n_slices)
        X = X.astype(np.float32)

        preds = model.predict(X)
        classes = np.argmax(preds, axis=1).reshape((n_slices_h, n_slices_v))

        cv2.imshow("img", frame)
        cv2.imshow("depth", depth.astype(np.uint8))
        print(classes)

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
