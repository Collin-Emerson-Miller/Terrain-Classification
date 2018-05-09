# import the necessary modules
from __future__ import division

import argparse
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

import cv2
import numpy as np
import os
from keras.models import model_from_json
from utils import prepare_images, class_label_overlay
import tensorflow as tf
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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



        self.colors = [(255, 255, 255), (0, 128, 0)]

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

        print(cv_image.shape)

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

            assignment_mask = cv2.resize(assignment_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            overlay = class_label_overlay(cv_image, assignment_mask, mask_opacity=0.3)


            cv2.imshow("img", cv_image)
            cv2.imshow("overlay", overlay)
            print(labels)

        cv2.imshow("img", cv_image)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
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



