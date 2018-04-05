# import the necessary modules
import freenect
import cv2
import numpy as np
import os
import uuid

CORPUS_PATH = "terrain_images"
GRASS_TERRAIN = "grass"
CONCRETE_TERRAIN = "concrete"


# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from kinect
def get_depth():
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array


if __name__ == "__main__":

    if not os.path.exists(os.path.join(CORPUS_PATH)):
        os.mkdir(CORPUS_PATH)

    if not os.path.exists(os.path.join(CORPUS_PATH, CONCRETE_TERRAIN)):
        os.mkdir(os.path.join(CORPUS_PATH, CONCRETE_TERRAIN))

    while 1:
        # get a frame from RGB camera
        frame = get_video()
        # get a frame from depth sensor
        depth = get_depth()

        cv2.imwrite(os.path.join(CORPUS_PATH, CONCRETE_TERRAIN, "{}.png".format(uuid.uuid4())), frame)

        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # display RGB image
        cv2.imshow('RGB image', frame)
        # display depth image
        cv2.imshow('Depth image', depth)

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
