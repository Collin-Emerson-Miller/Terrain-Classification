# import the necessary modules
from __future__ import division

import argparse
import freenect
import os



import cv2
import numpy as np
import pygame
import itertools
from OpenGL.GL import *
from OpenGL.GLU import *
from keras.applications import InceptionV3
from keras.layers import Input

import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model_name", help="Specify the name of the model to train.", default="inceptionV3")
parser.add_argument("weight_path", help="The directory to save the weights of the model.", default="weights")

args = parser.parse_args()

MODEL_NAME = args.model_name.lower()
WEIGHT_PATH = args.weight_path

print(WEIGHT_PATH)

# if os.path.exists(WEIGHT_PATH):
#     raise ValueError("Path to weights does not exist.  Did you provide the correct path?")


# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    return array


# function to get depth image from kinect
def get_depth():
    array, _ = freenect.sync_get_depth()
    return array


def Canonicalize(Depth):
    Aspect = Depth.shape[0] / Depth.shape[1]
    return [
        ((X / Depth.shape[0]) * Aspect - 0.5 * Aspect, Y / Depth.shape[1] - 0.5,Depth[X, Y])
        for X in range(Depth.shape[0])
        for Y in range(Depth.shape[1])
    ]


def Origin():
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(1, 0, 0)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 1, 0)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 1)
    glEnd()
    return


def frame_init(rgb, depth):
    Color = np.swapaxes(rgb, 0, 1)
    Color = np.flip(Color, 1)
    Color = np.divide(Color, 0xFF)  # Normalize into [0,1] range

    Depth = np.swapaxes(depth, 0, 1)
    Depth = np.flip(Depth,1)
    Depth = np.divide(Depth, 0xFFFF)  # Normalize into [0,1] range

    PointCloud = Canonicalize(Depth)
    PointCloud = list(np.asarray(PointCloud,dtype=np.float32).flatten())
    PointColor = list(Color.flatten())

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER,
                 len(PointCloud) * 4,
                 (ctypes.c_float * len(PointCloud))(*PointCloud),
                 GL_DYNAMIC_DRAW)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, cbo)
    glBufferData(GL_ARRAY_BUFFER,
                 len(PointColor) * 4,
                 (ctypes.c_float * len(PointColor))(*PointColor),
                 GL_DYNAMIC_DRAW)
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, None)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDrawArrays(GL_POINTS, 0, len(PointCloud))
    Origin()
    pygame.display.flip()

# Run once
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)

glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)

# Camera
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (display[0] / display[1]), 0.01, 100.0)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
gluLookAt(1.5, 0.5, 1.5,
          -0.5, -0.5, -0.5,
          0.0, 1.0, 0.0)

# Mesh
vbo = glGenBuffers(1)
cbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glEnableClientState(GL_VERTEX_ARRAY)
glVertexPointer(3, GL_FLOAT, 0, None)

glBindBuffer(GL_ARRAY_BUFFER, cbo)
glEnableClientState(GL_COLOR_ARRAY)
glColorPointer(3, GL_FLOAT, 0, None)

if __name__ == "__main__":

    n_slices = 2
    ratio = (4, 3)
    image_size = (640, 480)

    height = n_slices * ratio[1]
    width = n_slices * ratio[0]

    slice_height = int(image_size[1] / height)
    slice_width = int(image_size[0] / width)
    #
    # input_tensor = Input((slice_height, slice_width, 3), dtype=np.float32)
    #
    # model = InceptionV3(classes=2, weights=None, input_tensor=input_tensor)

    # print("Compiling Model...")
    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # print("Loading Model Weights...")
    # model.load_weights(WEIGHT_PATH)
    # print("Weights Loaded.")

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()
        #
        # X = utils.prepare_images(frame, image_size, ratio, n_slices)
        # X = X.astype(np.float32)
        #
        # preds = model.predict(X)
        # classes = np.argmax(preds, axis=1).reshape((height, width))

        frame_init(frame.astype(np.float32), depth.astype(np.float32))

        cv2.imshow("img", frame)
        cv2.imshow("depth", depth.astype(np.uint8))
        # print(classes)

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
