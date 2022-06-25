from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv3D, Concatenate


class DenseBlock(layers.Layer):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.layers = []

    def __conv__(self, output_channel, activation='relu'):
        return Conv2D(output_channel, (1, 1), activation=activation, padding="SAME")

    def __atrous_conv__(self, output_channel, dilation_rate=6, activation='relu'):
        return Conv2D(output_channel, (3, 3), dilation_rate=dilation_rate, activation=activation, padding="SAME")

    def call(self, inputs):

        k = int(inputs.shape[-1])
        dilation_rate = 6 * (k // 64)
        assert k % 8 == 0

        # l=1
        layer00 = inputs
        layer01 = self.__atrous_conv__(k, dilation_rate=dilation_rate)(layer00)
        layer02 = self.__atrous_conv__(k // 8, dilation_rate=dilation_rate)(layer01)

        # l=2
        layer10 = Concatenate(axis=-1)([layer00, layer02])
        layer11 = self.__conv__(k)(layer10)
        layer12 = self.__atrous_conv__(k, dilation_rate=dilation_rate)(layer11)
        layer13 = self.__atrous_conv__(k // 8, dilation_rate=dilation_rate)(layer12)

        # l=3
        layer20 = Concatenate(axis=-1)([layer00, layer02, layer13])
        layer21 = self.__conv__(k)(layer20)
        layer22 = self.__atrous_conv__(k, dilation_rate=dilation_rate)(layer21)
        layer23 = self.__atrous_conv__(k // 8, dilation_rate=dilation_rate)(layer22)

        # l=4
        layer30 = Concatenate(axis=-1)([layer00, layer02, layer13, layer23])
        layer31 = self.__conv__(k)(layer30)
        layer32 = self.__atrous_conv__(k, dilation_rate=dilation_rate)(layer31)
        layer33 = self.__atrous_conv__(k, dilation_rate=dilation_rate)(layer32)

        self.layers.extend([
            layer00, layer01, layer02,
            layer10, layer11, layer12, layer13,
            layer20, layer21, layer22, layer23,
            layer30, layer31, layer32, layer33,
        ])

        return layer33


class DenseBlock3D(layers.Layer):
    def __init__(self):
        super(DenseBlock3D, self).__init__()
        self.layers = []

    def __conv__(self, output_channel, activation='relu'):
        return Conv3D(output_channel, 3, activation=activation, padding="SAME")

    def __conv1__(self, output_channel, activation='relu'):
        return Conv3D(output_channel, 1, activation=activation, padding="SAME")

    def call(self, inputs):
        layer1 = self.__conv__(38)(inputs)

        layer2_1 = self.__conv__(38)(layer1)
        layer2_2 = layer1
        layer2 = Concatenate(axis=-1)([layer2_1, layer2_2])

        layer3_1 = self.__conv__(38)(layer2)
        layer3_2 = self.__conv1__(38)(layer2)
        layer3_3 = layer1
        layer3 = Concatenate(axis=-1)([layer3_1, layer3_2, layer3_3])

        layer4_1 = self.__conv__(38)(layer3)
        layer4_2 = self.__conv1__(38)(layer3)
        layer4_3 = self.__conv1__(38)(layer2)
        layer4_4 = layer1
        layer4 = Concatenate(axis=-1)([layer4_1, layer4_2, layer4_3, layer4_4])

        layer5_1 = self.__conv__(38)(layer4)
        layer5_2 = self.__conv1__(38)(layer4)
        layer5_3 = self.__conv1__(38)(layer3)
        layer5_4 = self.__conv1__(38)(layer2)
        layer5_5 = layer1
        layer5 = Concatenate(axis=-1)([layer5_1, layer5_2, layer5_3, layer5_4, layer5_5])

        return layer5

