import numpy
from PIL import Image

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from ctypes import *

import itertools

Color = numpy.array(Image.open("frame-000857.color.png"))
Color = numpy.transpose(Color,[1,0,2])
Color = numpy.flip(Color,1) # Flip Y
Color = numpy.divide(Color,0xFF) # Normalize into [0,1] range

Depth = numpy.array(Image.open("frame-000857.depth.png"))
Depth = numpy.transpose(Depth)
Depth = numpy.flip(Depth,1) # Flip Y
Depth = numpy.divide(Depth,0xFFFF) # Normalize into [0,1] range

def Canonicalize(Depth):
	Aspect = Depth.shape[0] / Depth.shape[1]
	return [
		((X / Depth.shape[0]) * Aspect - 0.5 * Aspect ,Y / Depth.shape[1] - 0.5,Depth[X,Y])
		 for X in range(Depth.shape[0])
		 for Y in range(Depth.shape[1])
	]

PointCloud = Canonicalize(Depth)
PointCloud = list(itertools.chain(*PointCloud))
PointColor = list(Color.flatten())

def Origin():
	glBegin(GL_LINES)
	glColor3f(1,0,0)
	glVertex3f(0,0,0)
	glVertex3f(1,0,0)
	glColor3f(0,1,0)
	glVertex3f(0,0,0)
	glVertex3f(0,1,0)
	glColor3f(0,0,1)
	glVertex3f(0,0,0)
	glVertex3f(0,0,1)
	glEnd()
	return

def main():
	pygame.init()
	display = (800,600)
	pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

	glEnable(GL_DEPTH_TEST)
	glDepthFunc(GL_LESS)


	# Camera
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(45, (display[0] / display[1]), 0.01, 100.0)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	gluLookAt(1.5, 1.5, 1.5, -0.5, -0.5, -0.5, 0.0, 1.0, 0.0) 

	# Mesh
	vbo = glGenBuffers(1)
	glBindBuffer(GL_ARRAY_BUFFER, vbo)
	glBufferData(GL_ARRAY_BUFFER,
		len(PointCloud) * 4,
		(ctypes.c_float * len(PointCloud))(*PointCloud),
		GL_STATIC_DRAW)

	cbo = glGenBuffers(1)
	glBindBuffer(GL_ARRAY_BUFFER, cbo)
	glBufferData(GL_ARRAY_BUFFER,
		len(PointColor) * 4,
		(ctypes.c_float * len(PointColor))(*PointColor),
		GL_STATIC_DRAW)

	
	glBindBuffer(GL_ARRAY_BUFFER, vbo)
	glEnableClientState(GL_VERTEX_ARRAY)
	glVertexPointer(3, GL_FLOAT, 0, None)

	glBindBuffer(GL_ARRAY_BUFFER, cbo)
	glEnableClientState(GL_COLOR_ARRAY)
	glColorPointer(3, GL_FLOAT, 0, None)

	while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					quit()

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			glRotate(15 * (1 / 60),0,1,0)
			glDrawArrays(GL_POINTS, 0, len(PointCloud))
			Origin()
			pygame.display.flip()
			pygame.time.wait(16)


main()