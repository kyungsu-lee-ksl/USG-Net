import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv3D, Conv3DTranspose, AveragePooling3D, Flatten, Dense, LeakyReLU

from util.layers import DenseBlock, DenseBlock3D


class USGNet(tf.keras.Model):

    def __init__(self):
        super().__init__()

        # classification module
        self.denseBlock01 = DenseBlock()
        self.denseBlock02 = DenseBlock()
        self.denseBlock03 = DenseBlock()
        self.denseBlock04 = DenseBlock()

        self.denseBlock3D01 = DenseBlock3D()
        self.denseBlock3D02 = DenseBlock3D()
        self.denseBlock3D03 = DenseBlock3D()
        self.denseBlock3D04 = DenseBlock3D()
        self.denseBlock3D05 = DenseBlock3D()
        self.denseBlock3D06 = DenseBlock3D()
        self.denseBlock3D07 = DenseBlock3D()

        self.generatedImage = None


    # layers
    def __conv__(self, output_channel, filter_size=3, activation='relu'):
        return Conv2D(output_channel, (filter_size, filter_size), activation=activation, padding="SAME")

    def __conv3D__(self, output_channel, filter_size=3, activation='relu'):
        return Conv3D(output_channel, filter_size, activation=activation, padding="SAME")

    # models for Generative Adversarial Network (GAN)
    def generator(self, inputs, depth=10, p=5):
        x = Concatenate(axis=3)([tf.expand_dims(inputs, axis=3) for _ in range(0, depth - p)])
        x = Conv3DTranspose(x.shape[-1], p + 1, activation='relu', padding="valid")(x) # (B, H+p, W+p, D, C)

        layer1 = self.__conv3D__(38)(x)
        layer2 = self.denseBlock3D01(layer1)
        layer3 = self.denseBlock3D02(layer2)
        layer4 = self.denseBlock3D03(layer3)

        x = Concatenate(axis=-1)([layer4, self.denseBlock3D04(layer4)])
        x = Concatenate(axis=-1)([layer3, self.denseBlock3D05(x)])
        x = Concatenate(axis=-1)([layer2, self.denseBlock3D06(x)])
        x = Concatenate(axis=-1)([layer1, self.denseBlock3D07(x)])

        x = self.__conv3D__(1, filter_size=1)(x)  # (B, H+p, W+p, D, C=1)
        self.generatedImage = x

        return x

    # model_pipelines
    def dense_pipeline(self, inputs):
        x = self.denseBlock01(self.__conv__(64)(inputs))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.denseBlock02(self.__conv__(128)(x))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.denseBlock03(self.__conv__(256)(x))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.denseBlock04(self.__conv__(512)(x))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x

    def generator_pipeline(self, inputs, depth=10, p=5):
        generated_3d_prediction = self.generator(inputs=inputs, depth=depth, p=p)
        x = Conv3D(64, 6, activation='relu')(generated_3d_prediction)
        x = AveragePooling3D(pool_size=(1, 1, 10), padding='SAME')(x)
        x = tf.squeeze(x, axis=3)
        x = self.dense_pipeline(x)
        return x


    # basic functions
    def call(self, inputs):
        from_dense_pipeline = self.dense_pipeline(inputs)
        from_generator_pipeline = self.generator_pipeline(inputs)
        x = Flatten()(from_dense_pipeline)
        y = Flatten()(from_generator_pipeline)

        z = Concatenate(axis=-1)([x, y])
        z = Dense(1000)(z)
        z = Dense(1000)(z)

        z1 = Dense(2)(z)
        z2 = Dense(9)(z)

        return z1, z2

    def build_graph(self, input_shape):
        x = keras.Input(shape=input_shape[1:])
        return keras.Model(inputs=[x], outputs=self.call(x))

    def getGeneratedImage(self):
        return self.generatedImage


class Discriminator(tf.keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes


    def call(self, inputs):
        x = Conv2D(64, (4, 4), activation=None, padding="SAME")(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D()(x)
        x = Conv2D(128, (4, 4), activation=None, padding="SAME")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D()(x)
        x = Conv2D(256, (4, 4), activation=None, padding="SAME")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D()(x)
        x = Conv2D(256, (4, 4), activation=None, padding="SAME")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(1000, activation='relu')(x)
        x = Dense(self.num_classes)(x)
        return x


    def build_graph(self, input_shape):
        x = keras.Input(shape=input_shape[1:])
        return keras.Model(inputs=[x], outputs=self.call(x))
