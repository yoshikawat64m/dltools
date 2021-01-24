"""Keras implementation of SSD."""
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Activation,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    Reshape,
    ZeroPadding2D
)

from ssd_layers import Normalize, PriorBox


class ConvLayers2D(Model):

    def __init__(self, layers, filters, kernel_size=3, pool=True, pool_size=(2, 2), pool_strides=(2, 2)):
        self._layers = []

        for i in range(layers):
            layer_name = 'conv{}'.format(i + 1, kernel_size, kernel_size)
            if type(filters) is int:
                layer = Conv2D(filters, kernel_size, activation="relu", padding='same')
            elif type(filters) in [list, tuple]:
                layer = Conv2D(filters[i], kernel_size, activation="relu", padding='same')
            else:
                raise Exception('Filters is invalid')
            setattr(self, layer_name, layer)
            self._layers.append(layer_name)

        if pool:
            layer_name = 'pool'
            layer = MaxPool2D(pool_size, strides=pool_strides, padding='same', name=layer_name)
            setattr(self, layer_name, layer)
            self._layers.append(layer_name)

    def call(self, x):
        outputs = {}
        for layer in self._layers:
            x = getattr(self, layer)(x)
            outputs[layer] = x
        return outputs


class SSD300(Model):
    """SSD300 architecture.
    Reference: https://arxiv.org/abs/1512.02325

    Params
    ------------
    input_shape: Shape of the input image,
        expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
    num_classes: Number of classes including background.
    """

    def __init__(self, input_shape, num_classes=21):
        super(SSD300, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        img_size = (self.input_shape[1], self.input_shape[0])

        # vgg16
        self.block1_conv_3x3x64 = ConvLayers2D(layers=2, filters=64, kernel_size=3, pool=True, name="block1_conv_3x3x64")
        self.block2_conv_3x3x128 = ConvLayers2D(layers=2, filters=128, kernel_size=3, pool=True, name="block2_conv_3x3x128")
        self.block3_conv_3x3x256 = ConvLayers2D(layers=3, filters=256, kernel_size=3, pool=True, name="block3_conv_3x3x256")
        self.block4_conv_3x3x512 = ConvLayers2D(layers=3, filters=512, kernel_size=3, pool=True, name="block4_conv_3x3x512")
        self.block5_conv_3x3x512 = ConvLayers2D(layers=3, filters=512, kernel_size=3, pool=True, pool_size=(3, 3), pool_strides=(1, 1), name="block5_conv_3x3x512")

        self.block6_conv_3x3x1024 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='block6_conv_3x3x1024')
        self.block6_conv_1x1x1024 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='block6_conv_1x1x1024')

        self.block7_conv_1x1x256 = Conv2D(256, (1, 1), activation='relu', padding='same', name='block7_conv_1x1x256')
        self.block7_conv_3x3x512 = Conv2D(512, (3, 3), subsample=(2, 2), activation='relu', padding='same', name='block7_conv_3x3x512')

        self.block8_conv_1x1x128 = Conv2D(128, (1, 1), activation='relu', padding='same', name='block8_conv_1x1x128')
        self.block8_conv_3x3x256 = Conv2D(256, (3, 3), subsample=(2, 2), activation='relu', padding='valid', name='block8_conv_3x3x256')

        self.block9_conv_1x1x128 = Conv2D(128, (1, 1), activation='relu', padding='same', name='block9_conv_1x1x128')
        self.block9_conv_3x3x256 = Conv2D(256, (3, 3), subsample=(2, 2), activation='relu', padding='same', name='block9_conv_3x3x256')

        self.block10_conv_1x1x128 = Conv2D(128, 1, 1, activation='relu', padding='same', name='block10_conv_1x1x128')
        self.block10_conv_3x3x256 = Conv2D(256, (3, 3), subsample=(2, 2), activation='relu', padding='same', name='block10_conv_3x3x256')

        self.flatten = Flatten()

        num_priors = 3
        self.block4_norm_mbox_loc = Conv2D(num_priors * 4, 3, 3, padding='same', name='block4_norm_mbox_loc')
        self.block4_norm_mbox_conf = Conv2D(num_priors * self.num_classes, 3, 3, padding='same', name='block4_norm_mbox_conf')
        self.block4_norm_mbox_priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='block4_norm_mbox_priorbox')
        self.block4_norm = Normalize(20, name='block4_norm')

        num_priors = 6
        self.block6_mbox_loc = Conv2D(num_priors * 4, 3, 3, padding='same', name='block6_mbox_loc')
        self.block6_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, padding='same', name='block6_mbox_conf')
        self.block6_mbox_priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='block6_mbox_priorbox')

        self.block7_mbox_loc = Conv2D(num_priors * 4, 3, 3, padding='same', name='block7_mbox_loc')
        self.block7_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, padding='same', name='block7_mbox_conf')
        self.block7_mbox_priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='block7_mbox_priorbox')

        self.block8_mbox_loc = Conv2D(num_priors * 4, 3, 3, padding='same', name='block8_mbox_loc')
        self.block8_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, padding='same', name='block8_mbox_conf')
        self.block8_mbox_priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='block8_mbox_priorbox')

        self.block9_mbox_loc = Conv2D(num_priors * 4, 3, 3, padding='same', name='block9_mbox_loc')
        self.block9_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, padding='same', name='block9_mbox_conf')
        self.block9_mbox_priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='block9_mbox_priorbox')

        self.block10_mbox_loc_flat = Dense(num_priors * 4, name='block10_mbox_loc_flat')
        self.block10_mbox_conf_flat = Dense(num_priors * num_classes, name='block10_mbox_conf_flat')
        self.block10_reshape = Reshape((1, 1, 256), name='block10_reshape')
        self.block10_mbox_priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='block10_mbox_priorbox')

        self.concat_conf = Concatenate(axis=1, name='mbox_conf')
        self.concat_loc = Concatenate(axis=1, name='mbox_loc')
        self.concat_priorbox = Concatenate(axis=1, name='mbox_priorbox')
        self.concat_predictions = Concatenate(axis=2, name='mbox_predictions')

        self.reshape_loc = Reshape((4, 4), name='mbox_loc_final')
        self.reshape_conf = Reshape((4, self.num_classes), name='mbox_conf_logits')

        self.activate_softmax = Activation('softmax', name='mbox_conf_softmax')

    def call(self, x):
        """SSD300 architecture.
        # Arguments
            input_shape: Shape of the input image,
                expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        # References
            https://arxiv.org/abs/1512.02325
        """
        # conv layers
        out1_conv = self.block1_conv_3x3x64(x)
        out2_conv = self.block2_conv_3x3x128(out1_conv['pool'])
        out3_conv = self.block3_conv_3x3x256(out2_conv['pool'])
        out4_conv = self.block4_conv_3x3x512(out3_conv['pool'])
        out5_conv = self.block5_conv_3x3x512(out4_conv['pool'])

        out6_conv = self.block6_conv_3x3x1024(out5_conv['pool'])
        out6_conv = self.block6_conv_1x1x1024(out6_conv)

        out7_conv = self.block7_conv_1x1x256(out6_conv)
        out7_conv = self.block7_conv_3x3x512(out7_conv)

        out8_conv = ZeroPadding2D(self.block8_conv_1x1x128(out7_conv))
        out8_conv = self.block8_conv_3x3x256(out8_conv)

        out9_conv = self.block9_conv_1x1x128(out8_conv)
        out9_conv = self.block9_conv_3x3x256(out9_conv)

        out10_conv = self.block10_conv_1x1x128(out9_conv)
        out10_conv = self.block10_conv_3x3x256(out10_conv)

        # Prediction from out4
        out4_conv3_norm = self.block4_norm(out4_conv['conv3'])
        out4_norm_mbox_loc_flat = self.flatten(self.block4_norm_mbox_loc(out4_conv3_norm))
        out4_norm_mbox_conf_flat = self.flatten(self.block4_norm_mbox_conf(out4_conv3_norm))
        out4_norm_mbox_priorbox = self.block4_norm_mbox_priorbox(out4_conv3_norm)

        # Prediction from out6
        out6_mbox_loc_flat = self.flatten(self.block6_mbox_loc(out6_conv))
        out6_mbox_conf_flat = self.flatten(self.block6_mbox_conf(out6_conv))
        out6_mbox_priorbox = self.block6_mbox_priorbox(out6_conv)

        # Prediction from out7
        out7_mbox_loc_flat = self.flatten(self.block7_mbox_loc(out7_conv))
        out7_mbox_conf_flat = self.flatten(self.block7_mbox_conf(out7_conv))
        out7_mbox_priorbox = self.block7_mbox_priorbox(out7_conv)

        # Prediction from out8
        out8_mbox_loc_flat = self.flatten(self.block8_mbox_loc(out8_conv))
        out8_mbox_conf_flat = self.flatten(self.block8_mbox_conf(out8_conv))
        out8_mbox_priorbox = self.block8_mbox_priorbox(out8_conv)

        # Prediction from out9
        out9_mbox_loc_flat = self.flatten(self.block9_mbox_loc(out9_conv))
        out9_mbox_conf_flat = self.flatten(self.block9_mbox_conf(out9_conv))
        out9_mbox_priorbox = self.block9_mbox_priorbox(out9_conv)

        # Prediction from out10
        out10_mbox_loc_flat = self.block10_mbox_loc_flat(out10_conv)
        out10_mbox_conf_flat = self.block10_mbox_conf_flat(out10_conv)
        out10_mbox_priorbox = self.block10_mbox_priorbox(self.block10_reshape(out10_conv))

        # Concatenate predictions
        mbox_loc = self.concat_loc([
            out4_norm_mbox_loc_flat,
            out6_mbox_loc_flat,
            out7_mbox_loc_flat,
            out8_mbox_loc_flat,
            out9_mbox_loc_flat,
            out10_mbox_loc_flat
        ])

        mbox_conf = self.concat_conf([
            out4_norm_mbox_conf_flat,
            out6_mbox_conf_flat,
            out7_mbox_conf_flat,
            out8_mbox_conf_flat,
            out9_mbox_conf_flat,
            out10_mbox_conf_flat
        ])

        mbox_priorbox = self.concat_priorbox([
            out4_norm_mbox_priorbox,
            out6_mbox_priorbox,
            out7_mbox_priorbox,
            out8_mbox_priorbox,
            out9_mbox_priorbox,
            out10_mbox_priorbox
        ])

        mbox_loc = self.reshape_loc(mbox_loc)
        mbox_conf = self.activate_softmax(self.reshape_conf(mbox_conf))

        predictions = self.concat_predictions([
            mbox_loc,
            mbox_conf,
            mbox_priorbox
        ])

        return predictions
