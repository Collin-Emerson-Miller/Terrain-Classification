from keras.layers import Input
from keras.applications.resnet50 import ResNet50


def ResNet50(input_shape, n_classes):
    input_tensor = Input(input_shape)
    return ResNet50(classes=n_classes, weights=None, input_tensor=input_tensor)
