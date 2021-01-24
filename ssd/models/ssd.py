"""Keras implementation of SSD."""
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Activation,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
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

        self.layer1_conv_3x3x64 = ConvLayers2D(layers=2, filters=64, pool=True, name="layer1")
        self.layer2_conv_3x3x128 = ConvLayers2D(layers=2, filters=128, pool=True, name="layer2")
        self.layer3_conv_3x3x256 = ConvLayers2D(layers=3, filters=256, pool=True, name="layer3")
        self.layer4_conv_3x3x512 = ConvLayers2D(layers=3, filters=512, pool=True, name="layer4")
        self.layer5_conv_3x3x512 = ConvLayers2D(layers=3, filters=512, pool=True, pool_size=(3, 3), pool_strides=(1, 1), name="layer5")
        self.layer6_conv_3x3x1024 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='layer6')
        self.layer7_conv_1x1x1024 = Conv2D(1024, (1, 1), activation='relu', border_mode='same', name='layer7')
        self.layer8_conv_1x1x256 = Conv2D(256, 1, 1, activation='relu', padding='same', name='layer8')
        self.layer9_conv_3x3x512 = Conv2D(512, 3, 3, subsample=(2, 2), activation='relu', padding='same', name='layer9')
        self.layer10_conv_1x1x128 = Conv2D(128, 1, 1, activation='relu', border_mode='same', name='layer10')
        self.layer11_conv_3x3x256 = Conv2D(256, 3, 3, subsample=(2, 2), activation='relu', border_mode='valid', name='layer11')
        self.layer12_conv_1x1x128 = Conv2D(128, 1, 1, activation='relu', border_mode='same', name='layer12')
        self.layer13_conv_3x3x256 = Conv2D(256, 3, 3, subsample=(2, 2), activation='relu', border_mode='same', name='layer13')
        self.layer14_pool2d = GlobalAveragePooling2D(name='layer14_pool2d')

        self.layer4_norm = Normalize(20, name='conv4_3_norm')

        self.flatten = Flatten()

        num_priors = 3
        self.layer4_norm_mbox_loc = Conv2D(3 * 4, 3, 3, border_mode='same', name='layer4_norm_mbox_loc')
        self.layer4_norm_mbox_conf = Conv2D(3 * self.num_classes, 3, 3, border_mode='same', name='layer4_norm_mbox_conf{}'.format(self.num_classes))
        self.layer4_norm_mbox_priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='layer4_norm_mbox_priorbox')

        num_priors = 6
        self.layer7_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='layer7_mbox_loc')
        self.layer7_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, border_mode='same', name='layer7_mbox_conf')
        self.layer7_mbox_priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='layer7_mbox_priorbox')

        self.layer9_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='layer9_mbox_loc')
        self.layer9_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, border_mode='same', name='layer9_mbox_conf')
        self.layer9_mbox_priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='layer9_mbox_priorbox')

        self.layer11_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='layer11_mbox_loc')
        self.layer11_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, border_mode='same', name='layer11_mbox_conf')
        self.layer11_mbox_priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='layer11_mbox_priorbox')

        self.layer13_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='layer13_mbox_loc')
        self.layer13_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, border_mode='same', name='layer13_mbox_conf')
        self.layer13_mbox_priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='layer13_mbox_priorbox')

        if K.image_dim_ordering() == 'tf':
            target_shape = (1, 1, 256)
        else:
            target_shape = (256, 1, 1)

        self.layer14_mbox_loc_flat = Dense(num_priors * 4, name='layer14_mbox_loc_flat')
        self.layer14_mbox_conf_flat = Dense(num_priors * num_classes, name='layer14_mbox_conf_flat')
        self.layer14_reshaped = Reshape(target_shape, name='layer14_reshaped')
        self.layer14_mbox_priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='layer14_mbox_priorbox')

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
        out1_conv = self.layer1_conv_3x3x64(x)
        out2_conv = self.layer2_conv_3x3x128(out1_conv['pool'])
        out3_conv = self.layer3_conv_3x3x256(out2_conv['pool'])
        out4_conv = self.layer4_conv_3x3x512(out3_conv['pool'])
        out5_conv = self.layer5_conv_3x3x512(out4_conv['pool'])

        out6_conv = self.layer6_conv_3x3x1024(out5_conv['pool'])
        out6_conv = self.layer7_conv_1x1x1024(out6_conv)

        out7_conv = self.layer8_conv_1x1x256(out6_conv)
        out7_conv = self.layer9_conv_3x3x512(out7_conv)

        out8_conv = ZeroPadding2D(self.layer10_conv_1x1x128(out7_conv))
        out8_conv = self.layer11_conv_3x3x256(out8_conv)

        out9_conv = self.layer12_conv_1x1x128(out8_conv)
        out9_conv = self.layer13_conv_3x3x256(out9_conv)

        out10_conv = self.layer12_conv_1x1x128(out9_conv)
        out10_conv = self.layer13_conv_3x3x256(out10_conv)

        # Prediction from out4
        out4_conv3_norm = self.layer4_norm(out4_conv['conv3'])
        out4_norm_mbox_loc_flat = self.flatten(self.layer4_norm_mbox_loc(out4_conv3_norm))
        out4_norm_mbox_conf_flat = self.flatten(self.layer4_norm_mbox_conf(out4_conv3_norm))
        out4_norm_mbox_priorbox = self.layer4_norm_mbox_priorbox(out4_conv3_norm)

        # Prediction from out6
        out7_mbox_loc_flat = self.flatten(self.layer7_mbox_loc(out6_conv))
        out7_mbox_conf_flat = self.flatten(self.layer7_mbox_conf(out6_conv))
        out7_mbox_priorbox = self.layer7_mbox_priorbox(out6_conv)

        # Prediction from out7
        out9_mbox_loc_flat = self.flatten(self.layer9_mbox_loc(out7_conv))
        out9_mbox_conf_flat = self.flatten(self.layer9_mbox_conf(out7_conv))
        out9_mbox_priorbox = self.layer9_mbox_priorbox(out7_conv)

        # Prediction from out8
        out11_mbox_loc_flat = self.flatten(self.layer11_mbox_loc(out8_conv))
        out11_mbox_conf_flat = self.flatten(self.layer11_mbox_conf(out8_conv))
        out11_mbox_priorbox = self.layer11_mbox_priorbox(out8_conv)

        # Prediction from out9
        out13_mbox_loc_flat = self.flatten(self.layer13_mbox_loc(out9_conv))
        out13_mbox_conf_flat = self.flatten(self.layer13_mbox_conf(out9_conv))
        out13_mbox_priorbox = self.layer13_mbox_priorbox(out9_conv)

        # Prediction from out10
        out14_mbox_loc_flat = self.layer14_mbox_loc_flat(out10_conv)
        out14_mbox_conf_flat = self.layer14_mbox_conf_flat(out10_conv)
        out14_mbox_priorbox = self.layer14_mbox_priorbox(self.layer14_reshaped(out10_conv))

        # Gather all predictions
        mbox_loc = self.concat_loc([
            out4_norm_mbox_loc_flat,
            out7_mbox_loc_flat,
            out9_mbox_loc_flat,
            out11_mbox_loc_flat,
            out13_mbox_loc_flat,
            out14_mbox_loc_flat
        ])

        mbox_conf = self.concat_conf([
            out4_norm_mbox_conf_flat,
            out7_mbox_conf_flat,
            out9_mbox_conf_flat,
            out11_mbox_conf_flat,
            out13_mbox_conf_flat,
            out14_mbox_conf_flat
        ])

        mbox_priorbox = self.concat_priorbox([
            out4_norm_mbox_priorbox,
            out7_mbox_priorbox,
            out9_mbox_priorbox,
            out11_mbox_priorbox,
            out13_mbox_priorbox,
            out14_mbox_priorbox
        ])

        mbox_loc = self.reshape_loc(mbox_loc)
        mbox_conf = self.activate_softmax(self.reshape_conf(mbox_conf))

        predictions = self.concat_predictions([
            mbox_loc,
            mbox_conf,
            mbox_priorbox
        ])

        return predictions
