from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3


def IncveptionV3(input_shape, n_classes):
    input_tensor = Input(input_shape)
    return InceptionV3(classes=n_classes, weights=None, input_tensor=input_tensor)
