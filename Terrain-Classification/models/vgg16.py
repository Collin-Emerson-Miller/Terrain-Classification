from keras.layers import Input
from keras.applications.vgg16 import VGG16


def Vgg16(input_shape, n_classes):
    input_tensor = Input(input_shape)
    return VGG16(classes=n_classes, weights=None, input_tensor=input_tensor)
