import tensorflow as tf
from tensorflow.keras import layers, models


def _conv_bn_relu(x, filters, kernel_size, strides=1, name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=None if name is None else name+"_conv")(x)
    x = layers.BatchNormalization(name=None if name is None else name+"_bn")(x)
    x = layers.Activation("relu", name=None if name is None else name+"_relu")(x)
    return x


def _basic_block(x, filters, stride=1, downsample=False, name=None):
    identity = x

    out = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                        kernel_initializer="he_normal", name=None if name is None else name+"_conv1")(x)
    out = layers.BatchNormalization(name=None if name is None else name+"_bn1")(out)
    out = layers.Activation("relu", name=None if name is None else name+"_relu1")(out)

    out = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False,
                        kernel_initializer="he_normal", name=None if name is None else name+"_conv2")(out)
    out = layers.BatchNormalization(name=None if name is None else name+"_bn2")(out)

    if downsample or identity.shape[-1] != filters:
        identity = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                 kernel_initializer="he_normal", name=None if name is None else name+"_down_conv")(identity)
        identity = layers.BatchNormalization(name=None if name is None else name+"_down_bn")(identity)

    out = layers.Add(name=None if name is None else name+"_add")([out, identity])
    out = layers.Activation("relu", name=None if name is None else name+"_out_relu")(out)
    return out


def _make_layer(x, filters, blocks, stride=1, name=None):
    x = _basic_block(x, filters, stride=stride, downsample=True, name=None if name is None else name+"_block0")
    for i in range(1, blocks):
        x = _basic_block(x, filters, stride=1, downsample=False, name=None if name is None else f"{name}_block{i}")
    return x


def _build_resnet(input_shape=(512, 512, 1), num_outputs=1, layers_cfg=(2, 2, 2, 2), name="resnet"):
    inputs = layers.Input(shape=input_shape)

    # Initial conv + maxpool
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, kernel_initializer="he_normal", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    # Residual layers
    x = _make_layer(x, 64,  layers_cfg[0], stride=1, name="layer1")
    x = _make_layer(x, 128, layers_cfg[1], stride=2, name="layer2")
    x = _make_layer(x, 256, layers_cfg[2], stride=2, name="layer3")
    x = _make_layer(x, 512, layers_cfg[3], stride=2, name="layer4")

    # Head for regression (1 output)
    x = layers.GlobalAveragePooling2D(name="avgpool")(x)
    outputs = layers.Dense(num_outputs, activation="linear", name="pred")(x)

    model = models.Model(inputs, outputs, name=name)
    return model


def build_resnet18(input_shape=(512, 512, 1), num_outputs=1):
    return _build_resnet(input_shape=input_shape, num_outputs=num_outputs, layers_cfg=(2, 2, 2, 2), name="resnet18")


def build_resnet34(input_shape=(512, 512, 1), num_outputs=1):
    return _build_resnet(input_shape=input_shape, num_outputs=num_outputs, layers_cfg=(3, 4, 6, 3), name="resnet34")

