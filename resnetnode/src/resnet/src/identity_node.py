#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import numpy as np


def listener():
    rospy.init_node('Identity_node')
    #image_topic = "/camera/color/image_raw"
    topic = "/terrain_scores_raw"
    rospy.Subscriber(topic, numpy_msg(Floats), callback)
    rospy.spin()

def callback(data):
    talker(data)

def talker(data):
    pub = rospy.Publisher('terrain_scores', numpy_msg(Floats),queue_size=10)
    r = rospy.Rate(10) # 10hz
    #print(data)
    #a = np.array(data, dtype=np.float32)
    while not rospy.is_shutdown():
        
        pub.publish(data)
        r.sleep()


if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
