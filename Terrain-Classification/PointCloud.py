# import the necessary modules
from __future__ import division

import cv2
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from utils import Canonicalize, get_depth, get_video


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
    # Depth = np.divide(Depth, 0xFFFF)  # Normalize into [0,1] range
    Depth = (Depth - Depth.min())/(Depth.max() - Depth.min())


    PointCloud = Canonicalize(Depth)
    PointCloud = list(np.array(PointCloud,dtype=np.float32).flatten())
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
gluLookAt(0.5, 0.5, 1.5,
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
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()

        cv2.imshow("Depth", depth.astype(np.uint8))

        frame_init(frame.astype(np.float32), depth.astype(np.float32))

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
