"""Keras implementation of SSD."""
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D


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