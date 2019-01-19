#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


import cv2


def talker():
    pub = rospy.Publisher('Imagepublisher', Image, queue_size=10)
    rospy.init_node('image', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    br = CvBridge()
    # dtype, n_channels = br.encoding_as_cvtype2('8UC3')
    # im = np.ndarray(shape=(224, 224, n_channels), dtype=dtype)
    while not rospy.is_shutdown():
        # msg = br.cv2_to_imgmsg(im)
        imgfile = cv2.imread('/home/phu/Documents/rock_1/rock_1_0001.jpg')
        im_mess = br.cv2_to_imgmsg(imgfile, encoding="passthrough")
        pub.publish(im_mess)
        rospy.loginfo(im_mess)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
