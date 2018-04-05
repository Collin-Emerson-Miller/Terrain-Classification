# import the necessary modules
import os
import uuid
import cv2

from utils import get_depth, get_video

CORPUS_PATH = "terrain_images"
GRASS_TERRAIN = "grass"
CONCRETE_TERRAIN = "concrete"


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
