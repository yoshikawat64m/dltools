"""Keras implementation of SSD."""
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from vgg.layers import ConvLayers2D


class VGG16(Model):
    """SSD300 architecture.
    Reference: https://arxiv.org/abs/1512.02325

    Params
    ------------
    input_shape: Shape of the input image,
        expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
    num_classes: Number of classes including background.

    References
    --------------------
    https://arxiv.org/pdf/1409.1556.pdf
    """

    def __init__(self, input_shape, num_classes=21):
        super(VGG16, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.block1_conv_3x3x64 = ConvLayers2D(layers=2, filters=64, kernel_size=3, pool=True, name="block1_conv_3x3x64")
        self.block2_conv_3x3x128 = ConvLayers2D(layers=2, filters=128, kernel_size=3, pool=True, name="block2_conv_3x3x128")
        self.block3_conv_3x3x256 = ConvLayers2D(layers=3, filters=256, kernel_size=3, pool=True, name="block3_conv_3x3x256")
        self.block4_conv_3x3x512 = ConvLayers2D(layers=3, filters=512, kernel_size=3, pool=True, name="block4_conv_3x3x512")
        self.block5_conv_3x3x512 = ConvLayers2D(layers=3, filters=512, kernel_size=3, pool=True, pool_size=(3, 3), pool_strides=(1, 1), name="block5_conv_3x3x512")

        self.flatten = Flatten(name='flatten')
        self.fc1_4096 = Dense(4096, activation='relu', name='fc1_4096')
        self.fc2_4096 = Dense(4096, activation='relu', name='fc2_4096')
        self.fc3_1000 = Dense(1000, activation='softmax', name='fc3_1000')

    def call(self, x):
        out1_conv = self.block1_conv_3x3x64(x)
        out2_conv = self.block2_conv_3x3x128(out1_conv['pool'])
        out3_conv = self.block3_conv_3x3x256(out2_conv['pool'])
        out4_conv = self.block4_conv_3x3x512(out3_conv['pool'])
        out5_conv = self.block5_conv_3x3x512(out4_conv['pool'])
        out5_conv_flatten = self.flatten(out5_conv['pool'])
        out6_fc = self.fc1_4096(out5_conv_flatten)
        out7_fc = self.fc2_4096(out6_fc)
        out8_fc = self.fc3_1000(out7_fc)

        return out8_fc
