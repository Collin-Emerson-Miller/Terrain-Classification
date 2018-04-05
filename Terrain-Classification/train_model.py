#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import glob
import os
import json

import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model_name", help="Specify the name of the model to train.", default="inceptionV3")
parser.add_argument("corpus_path", help="The directory with the images to be trained on.")
parser.add_argument("--weight_path", help="The directory to save the weights of the model.", default="weights")
parser.add_argument("--weight_file_name", help="The filename of the model weight file.", default="current timestamp")
parser.add_argument("--n_slices", help="The number of slices for a training image.", default=2)
parser.add_argument("--ratio", help="The aspect ratio of the images.", default=(4, 3))
parser.add_argument("--image_size", help="The size of the images for the network."
                                         "If an image is not the specified size, it will"
                                         " be resized.", default=(640, 480))
parser.add_argument("--gray", help="Train network with gray-scale images.")

args = parser.parse_args()

from keras.layers import Input

MODEL_NAME = args.model_name.lower()
CORPUS_PATH = args.corpus_path
WEIGHT_PATH = args.weight_path
WEIGHT_FILE_NAME = args.weight_file_name
N_SLICES = args.n_slices
RATIO = args.ratio
IMAGE_SIZE = args.image_size
GRAY = args.gray

config_dict = {
    "model_name": MODEL_NAME,
    "weight_path": WEIGHT_PATH,
    "train_gray": GRAY
}


if not WEIGHT_PATH:
    WEIGHT_PATH = "models"

if not WEIGHT_FILE_NAME:
    WEIGHT_FILE_NAME = datetime.datetime.now().isoformat()

classes = os.listdir(CORPUS_PATH)

image_list = []
label_list = []

for i, c in enumerate(classes):
        terrain_path = os.path.join(CORPUS_PATH, c)
        images = glob.glob(os.path.join(terrain_path, "*.png"))
        image_list += images
        label_list += [i] * len(images)

config_dict["classes"] = dict(zip(classes, range(len(classes))))

with open("config.txt", "w") as f:
    f.write(json.dumps(config_dict))

g = utils.input_image_generator(image_list, label_list, IMAGE_SIZE, RATIO, N_SLICES)

a = next(g)[0][0]

input_tensor = Input(a.shape, dtype=a.dtype)

if MODEL_NAME == "vgg16":
    print("Using vgg16")
    from keras.applications.vgg16 import VGG16

    model = VGG16(classes=len(classes), weights=None, input_tensor=input_tensor)
elif MODEL_NAME == "resnet50":
    print("Using resnet50")
    from keras.applications.resnet50 import ResNet50

    model = ResNet50(classes=len(classes), weights=None, input_tensor=input_tensor)

else:
    print("Using inceptionV3")
    from keras.applications.inception_v3 import InceptionV3

    model = InceptionV3(classes=len(classes), weights=None, input_tensor=input_tensor)

print("Compiling Model...")
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print("Training...")
model.fit_generator(g, len(image_list))

print("Saving Model...")
# serialize weights to HDF5
if not os.path.exists(WEIGHT_PATH):
    os.mkdir(WEIGHT_PATH)

model.save_weights(os.path.join(WEIGHT_PATH, WEIGHT_FILE_NAME))
print("Saved model to disk")
