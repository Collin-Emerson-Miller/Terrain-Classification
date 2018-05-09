# import the necessary modules
from __future__ import division

import argparse
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

import cv2
import numpy as np
import yaml
import os
from keras.models import model_from_json
from utils import get_depth, get_video, prepare_images, class_label_overlay, get_spaced_colors, Canonicalize

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model_name", help="Specify the name of the model to train.", default="squeezenet")

args = parser.parse_args()

MODEL_NAME = args.model_name.lower()

print(MODEL_NAME)

clock = pygame.time.Clock()


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
    Depth = np.flip(Depth, 1)
    Depth = 3 * np.divide(Depth, 0b11111111111)  # Normalize into [0,1] range

    PointCloud = Canonicalize(Depth)
    PointCloud = list(np.array(PointCloud, dtype=np.float32).flatten())
    PointColor = list(Color.flatten())

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferSubData(GL_ARRAY_BUFFER,
                    0,
                    len(PointCloud) * 4,
                    (ctypes.c_float * len(PointCloud))(*PointCloud))
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, cbo)
    glBufferSubData(GL_ARRAY_BUFFER,
                    0,
                    len(PointColor) * 4,
                    (ctypes.c_float * len(PointColor))(*PointColor))
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, None)

    glMatrixMode(GL_MODELVIEW)
    glRotatef(10 * (clock.tick() / 1000), 0, 1, 0);

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
gluLookAt(2.5, 1.5, 2.5,
          -0.5, -0.5, -0.5,
          0.0, 1.0, 0.0)

# Mesh
vbo = glGenBuffers(1)
cbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER,
             640 * 480 * 3 * 4,
             None,
             GL_STREAM_DRAW)
glEnableClientState(GL_VERTEX_ARRAY)
glVertexPointer(3, GL_FLOAT, 0, None)

glBindBuffer(GL_ARRAY_BUFFER, cbo)
glBufferData(GL_ARRAY_BUFFER,
             640 * 480 * 3 * 4,
             None,
             GL_STREAM_DRAW)
glEnableClientState(GL_COLOR_ARRAY)
glColorPointer(3, GL_FLOAT, 0, None)

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

    colors = [(255, 255, 255), (0, 128, 0)]

    while True:

        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()

        X = prepare_images(frame, image_size, ratio, n_slices)
        X = X.astype(np.float32)

        preds = model.predict(X)
        labels = np.argmax(preds, axis=1).reshape((n_slices_h, n_slices_v))

        # Initialize the class assignment mask.
        assignment_mask = np.zeros((labels.shape[0], labels.shape[1], 3))

        # Fill class labels into assignment mask.
        for label in xrange(len(classes)):
            assignment_mask[np.isin(labels, label)] = colors[label]

        assignment_mask = cv2.resize(assignment_mask, image_size, interpolation=cv2.INTER_NEAREST)

        overlay = class_label_overlay(frame, assignment_mask, mask_opacity=0.3)

        overlay[np.isin(depth, xrange(1200))] = (0, 0, 0)

        # overlay = cv2.resize(overlay, (800, 800), interpolation=cv2.INTER_CUBIC)
        # depth = cv2.resize(depth, (800, 800), interpolation=cv2.INTER_CUBIC)

        frame_init(overlay.astype(np.float32), depth.astype(np.float32))

        cv2.imshow("img", frame)
        cv2.imshow("depth", depth.astype(np.uint8))
        cv2.imshow("overlay", overlay)
        print(labels)

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
