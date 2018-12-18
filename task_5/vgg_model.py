import tensorflow as tf
from tensorflow.python.keras import models


class VGGModel:

    def __init__(self):
        pass

    @staticmethod
    def get_layers():

        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in VGGModel.get_style_layers()]
        content_outputs = [vgg.get_layer(name).output for name in VGGModel.get_content_layers()]
        model_outputs = style_outputs + content_outputs

        return models.Model(vgg.input, model_outputs)

    @staticmethod
    def get_num_content_layers():
        return len(VGGModel.get_content_layers())

    @staticmethod
    def get_num_style_layers():
        return len(VGGModel.get_style_layers())

    @staticmethod
    def get_style_layers():
        return ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    @staticmethod
    def get_content_layers():
        return ['block5_conv2']
