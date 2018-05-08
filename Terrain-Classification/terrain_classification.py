# import the necessary modules
from __future__ import division

import argparse

import cv2
import numpy as np
from keras.applications import InceptionV3
from keras.layers import Input

from utils import get_depth, get_video, prepare_images, get_spaced_colors, class_label_overlay

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model_name", help="Specify the name of the model to train.", default="inceptionV3")
parser.add_argument("weight_path", help="The directory to save the weights of the model.", default="weights")

args = parser.parse_args()

MODEL_NAME = args.model_name.lower()
WEIGHT_PATH = args.weight_path

print(WEIGHT_PATH)

# if os.path.exists(WEIGHT_PATH):
#     raise ValueError("Path to weights does not exist.  Did you provide the correct path?")

n_classes = 2

if __name__ == "__main__":

    n_slices = 2
    ratio = (4, 3)
    image_size = (640, 480)

    height = n_slices * ratio[1]
    width = n_slices * ratio[0]

    slice_height = int(image_size[1] / height)
    slice_width = int(image_size[0] / width)

    input_tensor = Input((slice_height, slice_width, 3), dtype=np.float32)

    model = InceptionV3(classes=n_classes, weights=None, input_tensor=input_tensor)

    print("Compiling Model...")
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Loading Model Weights...")
    model.load_weights(WEIGHT_PATH)
    print("Weights Loaded.")

    colors = get_spaced_colors(n_classes)

    while True:

        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()

        X = prepare_images(frame, image_size, ratio, n_slices)
        X = X.astype(np.float32)

        preds = model.predict(X)
        classes = np.argmax(preds, axis=1).reshape((height, width))

        classes = np.rot90(classes)

        # Initialize the class assignment mask.
        assignment_mask = np.zeros((classes.shape[0], classes.shape[1], 3))

        # Fill class labels into assignment mask.
        for label in xrange(n_classes):
            assignment_mask[np.isin(classes, label)] = colors[label]

        assignment_mask = cv2.resize(assignment_mask, image_size, interpolation=cv2.INTER_NEAREST)

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
