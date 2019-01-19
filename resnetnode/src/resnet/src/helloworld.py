#!/usr/bin/env python
# license removed for brevity
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
import rospy
from sensor_msgs.msg import Image
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import numpy as np
from keras.applications.resnet50 import preprocess_input, decode_predictions
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import tensorflow as tf

location = os.path.dirname(os.path.realpath(__file__))

bridge = CvBridge()
# model = load_model()
graph = tf.get_default_graph()




def load():
    # load json and create model
    json_file = open(os.path.join(location, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(location, "model.h5"))
    return loaded_model

model = load()

def talker(np_image):

    pub = rospy.Publisher('terrain_scores_raw', numpy_msg(Floats), queue_size=10)

    rate = rospy.Rate(10) # 10hz
    with graph.as_default():
    	preds = model.predict(np_image)
    #print(preds[0][0])
    while not rospy.is_shutdown():
	a = np.array(preds[0], dtype=np.float32)
	b = np.tile(a, (224*224,1))
	b.reshape((224,224,6)).shape
	# rospy.loginfo(data)
	# cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
	# cv_image = cv_image[...,::-1]
	# new_image = image.load_img(data, target_size=(224, 224))
	# x = image.img_to_array(cv_image)
	# x = np.expand_dims(x, axis=0)
	# x = preprocess_input(x)
	# preds = model.predict(data)
	# pred_string = decode_predictions(preds, top=1000)[0]
	# print(pred_string)
	#rospy.loginfo(preds)
	pub.publish(b)
	rate.sleep()

def callback(image_msg):
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    cv_image = cv_image[...,::-1]
    cv_image = cv2.resize(cv_image, (224, 224))  # resize image
    np_image = np.asarray(cv_image)               # read as np array
    np_image = np.expand_dims(np_image, axis=0)   # Add another dimension for tensorflow
    np_image = np_image.astype(float)  # preprocess needs float64 and img is uint8
    np_image = preprocess_input(np_image)
    # preds = model.predict(np_image)
    talker(np_image)

def listener():
    rospy.init_node('image_sub')
    image_topic = "/camera/rgb/image_raw"
    #image_topic = "/Imagepublisher"
    rospy.Subscriber(image_topic, Image, callback)
    rospy.spin()




if __name__ == '__main__':
    try:
	listener()
    except rospy.ROSInterruptException:
	pass

# varS=None
#
# def fnc_callback(msg):
#     global varS
#     varS=msg.data
#
# def load_model():
#     # load json and create model
#     json_file = open('/home/phu/model.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights("/home/phu/model.h5")
#     return loaded_model
#
# if __name__=='__main__':
#     rospy.init_node('resnet')
#     image_topic = "/camera/image_raw"
#     sub=rospy.Subscriber(image_topic, Image, fnc_callback)
#     pub=rospy.Publisher('sub_pub', Int32, queue_size=1)
#     rate=rospy.Rate(10)
#
#     while not rospy.is_shutdown():
#         if varS<= 2500:
#             varP=0
#         else:
#              varP=1
#
#         pub.publish(varP)
#         rate.sleep()
