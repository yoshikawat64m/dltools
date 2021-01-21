"""Keras implementation of SSD."""
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dropout,
    Activation,
    AtrousConvolution2D,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    MaxPool2D,
    merge,
    Reshape,
    ZeroPadding2D
)

from ssd_layers import Normalize
from ssd_layers import PriorBox


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
        self.input_shape = input_shape
        self.num_classes = num_classes

        img_size = (self.input_shape[1], self.input_shape[0])

        super(SSD300, self).__init__()
        self.conv1_1 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1')
        self.conv1_2 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1')
        self.pool1 = MaxPool2D((2, 2), strides=(2, 2), border_mode='same', name='pool1')

        self.conv2_1 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1')
        self.conv2_2 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2')
        self.pool2 = MaxPool2D((2, 2), strides=(2, 2), border_mode='same', name='pool2')

        self.conv3_1 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')
        self.conv3_2 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')
        self.conv3_3 = Conv2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3')
        self.pool3 = MaxPool2D((2, 2), strides=(2, 2), border_mode='same', name='pool3')

        self.conv4_1 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1')
        self.conv4_2 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2')
        self.conv4_3 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3')
        self.pool4 = MaxPool2D((2, 2), strides=(2, 2), border_mode='same', name='pool4')

        self.conv5_1 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1')
        self.conv5_2 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2')
        self.conv5_3 = Conv2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3')
        self.pool5 = MaxPool2D((3, 3), strides=(1, 1), border_mode='same', name='pool5')

        self.fc6 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6), activation='relu', border_mode='same', name='fc6')
        self.fc7 = Conv2D(1024, 1, 1, activation='relu', border_mode='same', name='fc7')

        self.conv6_1 = Conv2D(256, 1, 1, activation='relu', border_mode='same', name='conv6_1')
        self.conv6_2 = Conv2D(512, 3, 3, subsample=(2, 2), activation='relu', border_mode='same', name='conv6_2')

        self.conv7_1 = Conv2D(128, 1, 1, activation='relu', border_mode='same', name='conv7_1')
        self.conv7_2 = Conv2D(256, 3, 3, subsample=(2, 2), activation='relu', border_mode='valid', name='conv7_2')

        self.conv8_1 = Conv2D(128, 1, 1, activation='relu', border_mode='same', name='conv8_1')
        self.conv8_2 = Conv2D(256, 3, 3, subsample=(2, 2), activation='relu', border_mode='same', name='conv8_2')

        self.norm4_3 = Normalize(20, name='conv4_3_norm')

        self.pool6 = GlobalAveragePooling2D(name='pool6')

        self.flatten = Flatten()

        num_priors = 3
        self.conv4_3_norm_mbox_loc = Conv2D(3 * 4, 3, 3, border_mode='same', name='conv4_3_norm_mbox_loc')
        self.conv4_3_norm_mbox_conf = Conv2D(3 * self.num_classes, 3, 3, border_mode='same', name='conv4_3_norm_mbox_conf_{}'.format(self.num_classes))
        self.conv4_3_norm_mbox_priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')

        num_priors = 6
        self.fc7_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='fc7_mbox_loc')
        self.fc7_mbox_conf = Conv2D(num_priors * num_classes, 3, 3,  border_mode='same', name='fc7_mbox_conf_{}'.format(self.num_classes))
        self.fc7_mbox_priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')

        self.conv6_2_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='conv6_2_mbox_loc')
        self.conv6_2_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, border_mode='same', name='conv6_2_mbox_conf')
        self.conv6_2_mbox_priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')

        self.conv7_2_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='conv7_2_mbox_loc')
        self.conv7_2_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, border_mode='same', name=name)
        self.conv7_2_mbox_priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')

        self.conv8_2_mbox_loc = Conv2D(num_priors * 4, 3, 3, border_mode='same', name='conv8_2_mbox_loc')
        self.conv8_2_mbox_conf = Conv2D(num_priors * num_classes, 3, 3, border_mode='same', name='conv8_2_mbox_conf')
        self.conv8_2_mbox_priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')

        if K.image_dim_ordering() == 'tf':
            target_shape = (1, 1, 256)
        else:
            target_shape = (256, 1, 1)

        self.pool6_mbox_loc_flat = Dense(num_priors * 4, name='pool6_mbox_loc_flat')
        self.pool6_mbox_conf_flat = Dense(num_priors * num_classes, name='pool6_mbox_conf_flat')
        self.pool6_mbox_priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')
        self.pool6_reshaped = Reshape(target_shape, name='pool6_reshaped')

        if hasattr(mbox_loc, '_keras_shape'):
            num_boxes = mbox_loc._keras_shape[-1] // 4
        elif hasattr(net['mbox_loc'], 'int_shape'):
            num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4

        self.mbox_loc_final = Reshape((num_boxes, 4), name='mbox_loc_final')
        self.mbox_conf_logits = Reshape((num_boxes, num_classes), name='mbox_conf_logits')
        self.mbox_conf_final = Activation('softmax', name='mbox_conf_final')

    def call(self, x):
        """SSD300 architecture.
        # Arguments
            input_shape: Shape of the input image,
                expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        # References
            https://arxiv.org/abs/1512.02325
        """
        # Block 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        out_1 = self.pool1(x)

        # Block 2
        x = self.conv2_1(out_1)
        x = self.conv2_2(x)
        out_2 = self.pool2(x)

        # Block 3
        x = self.conv3_1(out_2)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        out_3 = self.pool3(x)

        # Block 4
        x = self.conv4_1(out_3)
        x = self.conv4_2(x)
        out_conv4_3 = self.conv4_3(x)
        out_4 = self.pool4(x)

        # Block 5
        x = self.conv5_1(out_4)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        out_5 = self.pool5(x)

        x = self.fc6(out_5)
        x = Dropout(0.5, name='drop6')(x)
        out_fc7 = self.fc7(x)
        x = Dropout(0.5, name='drop7')(fc7)

        # Block 6
        x = self.conv6_1(x)
        out_conv6_2 = self.conv6_2(x)

        # Block 7
        x = ZeroPadding2D(self.conv7_1(out_conv6_2))
        x = self.conv7_2(x)

        # Block 8
        x = self.conv8_1(x)
        out_conv8_2 = self.conv8_2(x)

        # Last Pool
        out_pool6 = self.pool6(out_conv8_2)

        # Prediction from conv4_3
        out_conv4_3_norm = self.norm4_3(out_conv4_3)
        out_conv4_3_norm_mbox_loc_flat = self.flatten(self.conv4_3_norm_mbox_loc(out_conv4_3_norm))
        out_conv4_3_norm_mbox_conf = self.conv4_3_norm_mbox_conf(out_conv4_3_norm)
        out_conv4_3_norm_mbox_conf_flat = self.flatten(out_conv4_3_norm_mbox_conf)
        out_conv4_3_norm_mbox_priorbox = self.conv4_3_norm_mbox_priorbox(out_conv4_3_norm)

        # Prediction from fc7
        out_fc7_mbox_loc = self.fc7_mbox_loc(out_fc7)
        out_fc7_mbox_loc_flat = self.flatten(out_fc7_mbox_loc)
        out_fc7_mbox_conf = self.fc7_mbox_conf(out_fc7)
        out_fc7_mbox_conf_flat = self.flatten(out_fc7_mbox_conf)
        out_fc7_mbox_priorbox = self.fc7_mbox_priorbox(out_fc7)

        # Prediction from conv6_2
        out_conv6_2_mbox_loc = self.conv6_2_mbox_loc(out_conv6_2)
        out_conv6_2_mbox_loc_flat = self.flatten(out_conv6_2_mbox_loc)
        out_conv6_2_mbox_conf = self.conv6_2_mbox_conf(out_conv6_2)
        out_conv6_2_mbox_conf_flat = self.flatten(out_conv6_2_mbox_conf)
        out_conv6_2_mbox_priorbox = self.conv6_2_mbox_priorbox(out_conv6_2)

        # Prediction from conv7_2
        out_conv7_2_mbox_loc = self.conv7_2_mbox_loc(out_conv7_2)
        out_conv7_2_mbox_loc_flat = self.flatten(out_conv7_2_mbox_loc)
        out_conv7_2_mbox_conf = self.conv7_2_mbox_conf(out_conv7_2)
        out_conv7_2_mbox_conf_flat = self.flatten(out_conv7_2_mbox_conf)
        out_conv7_2_mbox_priorbox = self.conv7_2_mbox_priorbox(out_conv7_2)

        # Prediction from conv8_2
        out_conv8_2_mbox_loc = self.conv8_2_mbox_loc(out_conv8_2)
        out_conv8_2_mbox_loc_flat = self.flatten(out_conv8_2_mbox_loc)
        out_conv8_2_mbox_conf = self.conv8_2_mbox_conf(out_conv8_2)
        out_conv8_2_mbox_conf_flat = self.flatten(out_conv8_2_mbox_conf)
        out_conv8_2_mbox_priorbox = self.conv8_2_mbox_priorbox(out_conv8_2)

        # Prediction from pool6
        out_pool6_mbox_loc_flat = self.pool6_mbox_loc_flat(out_pool6)
        out_pool6_mbox_conf_flat = self.pool6_mbox_conf_flat(out_pool6)
        out_pool6_reshaped = self.pool6_reshaped(out_pool6)
        out_pool6_mbox_priorbox = self.pool6_mbox_priorbox(out_pool6_reshaped)

        # Gather all predictions
        mbox_loc = merge(
            [
                out_conv4_3_norm_mbox_loc_flat,
                out_fc7_mbox_loc_flat,
                out_conv6_2_mbox_loc_flat,
                out_conv7_2_mbox_loc_flat,
                out_conv8_2_mbox_loc_flat,
                out_pool6_mbox_loc_flat
            ],
            mode='concat',
            concat_axis=1,
            name='mbox_loc')

        mbox_conf = merge(
            [
                out_conv4_3_norm_mbox_conf_flat,
                out_fc7_mbox_conf_flat,
                out_conv6_2_mbox_conf_flat,
                out_conv7_2_mbox_conf_flat,
                out_conv8_2_mbox_conf_flat,
                out_pool6_mbox_conf_flat
            ],
            mode='concat',
            concat_axis=1,
            name='mbox_conf')

        mbox_priorbox = merge(
            [
                out_conv4_3_norm_mbox_priorbox,
                out_fc7_mbox_priorbox,
                out_conv6_2_mbox_priorbox,
                out_conv7_2_mbox_priorbox,
                out_conv8_2_mbox_priorbox,
                out_pool6_mbox_priorbox
            ],
            mode='concat',
            concat_axis=1,
            name='mbox_priorbox')

        mbox_loc = self.mbox_loc_final(mbox_loc)
        mbox_conf = self.mbox_conf_logits(mbox_conf)
        mbox_conf = self.mbox_conf_final(mbox_conf)

        predictions = merge(
            [
                mbox_loc,
                mbox_conf,
                mbox_priorbox
            ],
            mode='concat',
            concat_axis=2,
            name='predictions')

        return predictions
