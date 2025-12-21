import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Add,
    SpatialDropout2D,
)
from models.advanced_backbones import (
    build_squeezenet_backbone,
    build_vit_backbone,
    build_pvt_backbone,
    build_pvtv2_backbone,
    build_efficientformer_backbone,
    build_efficientvit_backbone,
)
try:
    # Optional small ResNet builders (18/34)
    from models.resnet_small import build_resnet18 as _build_resnet18_small, build_resnet34 as _build_resnet34_small
except Exception:
    _build_resnet18_small = None
    _build_resnet34_small = None


def _head_regression(x, dense_units=(1024, 1024), dropout_rate=None):
    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='linear', name='altitude_output')(x)
    return out


def _build_application_backbone(
    app_builder,
    input_shape,
    train_base,
    lr,
    weights,
    dense_units=(1024, 1024),
    dropout_rate=None,
    **builder_kwargs,
):
    base = app_builder(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
        **builder_kwargs,
    )
    for layer in base.layers:
        layer.trainable = bool(train_base)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    out = _head_regression(x, dense_units=dense_units, dropout_rate=dropout_rate)
    model = Model(base.input, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


def _build_from_backbone(backbone, train_base, lr, dense_units=(1024, 1024), dropout_rate=None):
    for layer in backbone.layers:
        layer.trainable = bool(train_base)
    x = backbone.output
    out = _head_regression(x, dense_units=dense_units, dropout_rate=dropout_rate)
    model = Model(backbone.input, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


def build_resnet50(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('resnet50 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.ResNet50,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
    )


def build_vgg16(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('vgg16 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.VGG16,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.5,
    )


def build_efficientnetb0(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('efficientnetb0 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.EfficientNetB0,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.5,
    )


def build_mobilenetv2(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('mobilenetv2 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.MobileNetV2,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(512, 256),
        dropout_rate=0.3,
    )


def build_mobilenetv3small(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('mobilenetv3small expects 3-channel input')
    try:
        builder = tf.keras.applications.MobileNetV3Small
    except AttributeError as e:
        raise ImportError('MobileNetV3Small is not available in this TensorFlow version') from e
    return _build_application_backbone(
        builder,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(512, 256),
        dropout_rate=0.3,
    )

def build_efficientnetv2b0(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('efficientnetv2b0 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.EfficientNetV2B0,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.5,
    )


def build_mobilenetv1(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('mobilenetv1 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.MobileNet,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(512, 256),
        dropout_rate=0.3,
    )


def build_mobilenetv3large(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('mobilenetv3large expects 3-channel input')
    try:
        builder = tf.keras.applications.MobileNetV3Large
    except AttributeError as e:
        raise ImportError('MobileNetV3Large is not available in this TensorFlow version') from e
    return _build_application_backbone(
        builder,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(640, 320),
        dropout_rate=0.4,
    )


def build_xception(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('xception expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.Xception,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.4,
    )


def build_densenet121(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('densenet121 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.DenseNet121,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.4,
    )


def build_densenet169(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('densenet169 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.DenseNet169,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.4,
    )


def build_densenet201(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    if input_shape[-1] != 3:
        raise ValueError('densenet201 expects 3-channel input')
    return _build_application_backbone(
        tf.keras.applications.DenseNet201,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.4,
    )


def _build_convnext_variant(variant, input_shape, train_base, lr, weights):
    if input_shape[-1] != 3:
        raise ValueError('convnext expects 3-channel input')
    try:
        builder = getattr(tf.keras.applications, f'ConvNeXt{variant}')
    except AttributeError as e:
        raise ImportError('ConvNeXt models are not available in this TensorFlow version') from e
    return _build_application_backbone(
        builder,
        input_shape=input_shape,
        train_base=train_base,
        lr=lr,
        weights=weights,
        dense_units=(1024, 512),
        dropout_rate=0.4,
        include_preprocessing=True,
    )


def build_convnext_tiny(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    return _build_convnext_variant('Tiny', input_shape, train_base, lr, weights)


def build_convnext_small(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    return _build_convnext_variant('Small', input_shape, train_base, lr, weights)


def build_convnext_base(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    return _build_convnext_variant('Base', input_shape, train_base, lr, weights)


def build_convnext_large(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights='imagenet'):
    return _build_convnext_variant('Large', input_shape, train_base, lr, weights)


def build_squeezenet(input_shape=(512, 512, 3), train_base=True, lr=5e-5, weights=None):
    if input_shape[-1] != 3:
        raise ValueError('squeezenet expects 3-channel input')
    backbone = build_squeezenet_backbone(input_shape)
    return _build_from_backbone(backbone, train_base=train_base, lr=lr, dense_units=(1024, 512), dropout_rate=0.4)


def _build_vit_model(variant, input_shape, train_base, lr):
    backbone = build_vit_backbone(variant, input_shape)
    return _build_from_backbone(backbone, train_base=train_base, lr=lr, dense_units=(1024, 512), dropout_rate=0.4)


def _build_pvt_model(variant, input_shape, train_base, lr, version='v1'):
    if version == 'v1':
        backbone = build_pvt_backbone(variant, input_shape)
    else:
        backbone = build_pvtv2_backbone(variant, input_shape)
    return _build_from_backbone(backbone, train_base=train_base, lr=lr, dense_units=(1024, 512), dropout_rate=0.4)


def _build_efficientformer_model(variant, input_shape, train_base, lr):
    backbone = build_efficientformer_backbone(variant, input_shape)
    return _build_from_backbone(backbone, train_base=train_base, lr=lr, dense_units=(1024, 512), dropout_rate=0.3)


def _build_efficientvit_model(variant, input_shape, train_base, lr):
    backbone = build_efficientvit_backbone(variant, input_shape)
    return _build_from_backbone(backbone, train_base=train_base, lr=lr, dense_units=(1024, 512), dropout_rate=0.3)


def build_custom_cnn(
    input_shape=(512, 512, 1),
    lr=1e-4,
    use_augment=True,
    huber_delta=50.0,
    se_ratio=0.25,
):
    """
    ResNet-inspired, bottleneck-based custom CNN for altitude regression.

    Differences from canonical ResNet-50 (keeps originality):
    - Stem is triple 3x3 convs + maxpool (no 7x7)
    - Bottleneck blocks (expansion=4) with optional SE attention
    - Stride applied in 3x3 (ResNet-D style)
    - Lite widths and mild SpatialDropout in deeper stages
    - Huber regression head
    """

    wd = tf.keras.regularizers.l2(1e-4)

    def se_block(x, ratio=0.25):
        c = int(x.shape[-1]) if x.shape[-1] is not None else 0
        if c == 0:
            return x
        s = tf.keras.layers.GlobalAveragePooling2D()(x)
        s = tf.keras.layers.Dense(max(1, int(c * ratio)), activation='relu', kernel_regularizer=wd)(s)
        s = tf.keras.layers.Dense(c, activation='sigmoid', kernel_regularizer=wd)(s)
        s = tf.keras.layers.Reshape((1, 1, c))(s)
        return tf.keras.layers.Multiply()([x, s])

    def bottleneck(x, filters, stride=1, expansion=4, use_se=True, drop_rate=0.0):
        in_ch = int(x.shape[-1]) if x.shape[-1] is not None else None
        shortcut = x

        y = BatchNormalization()(x)
        y = tf.keras.layers.ReLU()(y)
        y = Conv2D(filters, 1, strides=1, padding='same', use_bias=False, kernel_regularizer=wd, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        y = Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, kernel_regularizer=wd, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        y = Conv2D(filters * expansion, 1, strides=1, padding='same', use_bias=False, kernel_regularizer=wd, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)

        if use_se:
            y = se_block(y, ratio=se_ratio)

        if stride != 1 or (in_ch is not None and in_ch != filters * expansion):
            shortcut = Conv2D(filters * expansion, 1, strides=stride, padding='same', use_bias=False, kernel_regularizer=wd, kernel_initializer='he_normal')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        out = Add()([shortcut, y])
        if drop_rate and drop_rate > 0:
            out = SpatialDropout2D(drop_rate)(out)
        return out

    def make_stage(x, filters, blocks, stride, drop_rate):
        x = bottleneck(x, filters, stride=stride, expansion=4, use_se=True, drop_rate=drop_rate)
        for _ in range(blocks - 1):
            x = bottleneck(x, filters, stride=1, expansion=4, use_se=True, drop_rate=drop_rate)
        return x

    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    if use_augment:
        aug_layers = []
        try:
            aug_layers.append(tf.keras.layers.RandomFlip('horizontal'))
        except Exception:
            pass
        try:
            aug_layers.append(tf.keras.layers.RandomRotation(0.05))
        except Exception:
            pass
        try:
            aug_layers.append(tf.keras.layers.RandomZoom(0.1))
        except Exception:
            pass
        try:
            aug_layers.append(tf.keras.layers.RandomContrast(0.1))
        except Exception:
            pass
        if aug_layers:
            x = tf.keras.Sequential(aug_layers, name='augment')(x)

    # Distinct stem: 3x3 x3 + max pool
    x = Conv2D(64, 3, strides=2, padding='same', use_bias=False, kernel_regularizer=wd, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, kernel_regularizer=wd, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(96, 3, strides=1, padding='same', use_bias=False, kernel_regularizer=wd, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stages (lite widths): (filters, blocks, stride, dropout)
    for f, n, s, dr in [(32, 2, 1, 0.00), (64, 3, 2, 0.05), (96, 4, 2, 0.10), (128, 2, 2, 0.10)]:
        x = make_stage(x, f, blocks=n, stride=s, drop_rate=dr)

    # Head
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=wd)(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation='relu', kernel_regularizer=wd)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='linear', name='altitude_output')(x)

    model = Model(inputs, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.Huber(delta=huber_delta)
    model.compile(optimizer=opt, loss=loss, metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model


def get_model(arch: str, input_shape=(512, 512, 3), lr=5e-5, train_base=True, weights='imagenet'):
    arch = (arch or 'resnet50').lower()
    if arch == 'squeezenet':
        return build_squeezenet(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('vit_t', 'vit_tiny', 'vit-tiny'):
        return _build_vit_model('vit_tiny', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('vit_s', 'vit_small', 'vit-small'):
        return _build_vit_model('vit_small', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('vit_b', 'vit_base', 'vit-base'):
        return _build_vit_model('vit_base', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('vit_l', 'vit_large', 'vit-large'):
        return _build_vit_model('vit_large', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('pvt_tiny', 'pvt-tiny'):
        return _build_pvt_model('pvt_tiny', input_shape=input_shape, train_base=train_base, lr=lr, version='v1')
    if arch in ('pvt_small', 'pvt-s'):
        return _build_pvt_model('pvt_small', input_shape=input_shape, train_base=train_base, lr=lr, version='v1')
    if arch in ('pvt_medium', 'pvt-m'):
        return _build_pvt_model('pvt_medium', input_shape=input_shape, train_base=train_base, lr=lr, version='v1')
    if arch in ('pvt_large', 'pvt-l'):
        return _build_pvt_model('pvt_large', input_shape=input_shape, train_base=train_base, lr=lr, version='v1')
    if arch in ('pvtv2_b0', 'pvtv2-b0'):
        return _build_pvt_model('pvtv2_b0', input_shape=input_shape, train_base=train_base, lr=lr, version='v2')
    if arch in ('pvtv2_b1', 'pvtv2-b1'):
        return _build_pvt_model('pvtv2_b1', input_shape=input_shape, train_base=train_base, lr=lr, version='v2')
    if arch in ('pvtv2_b2', 'pvtv2-b2'):
        return _build_pvt_model('pvtv2_b2', input_shape=input_shape, train_base=train_base, lr=lr, version='v2')
    if arch in ('pvtv2_b3', 'pvtv2-b3'):
        return _build_pvt_model('pvtv2_b3', input_shape=input_shape, train_base=train_base, lr=lr, version='v2')
    if arch in ('pvtv2_b4', 'pvtv2-b4'):
        return _build_pvt_model('pvtv2_b4', input_shape=input_shape, train_base=train_base, lr=lr, version='v2')
    if arch in ('pvtv2_b5', 'pvtv2-b5'):
        return _build_pvt_model('pvtv2_b5', input_shape=input_shape, train_base=train_base, lr=lr, version='v2')
    if arch in ('efficientformer_l1', 'effformer_l1'):
        return _build_efficientformer_model('efficientformer_l1', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientformer_l3', 'effformer_l3'):
        return _build_efficientformer_model('efficientformer_l3', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientformer_l7', 'effformer_l7'):
        return _build_efficientformer_model('efficientformer_l7', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientvit_m0', 'evit_m0'):
        return _build_efficientvit_model('efficientvit_m0', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientvit_m1', 'evit_m1'):
        return _build_efficientvit_model('efficientvit_m1', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientvit_m2', 'evit_m2'):
        return _build_efficientvit_model('efficientvit_m2', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientvit_m3', 'evit_m3'):
        return _build_efficientvit_model('efficientvit_m3', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientvit_m4', 'evit_m4'):
        return _build_efficientvit_model('efficientvit_m4', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch in ('efficientvit_m5', 'evit_m5'):
        return _build_efficientvit_model('efficientvit_m5', input_shape=input_shape, train_base=train_base, lr=lr)
    if arch == 'resnet50':
        return build_resnet50(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('vgg16', 'vgg'):
        return build_vgg16(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('efficientnetv2b0', 'effnetv2b0'):
        return build_efficientnetv2b0(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('efficientnetb0', 'effnetb0'):
        return build_efficientnetb0(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('xception',):
        return build_xception(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('densenet121', 'dn121'):
        return build_densenet121(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('densenet169', 'dn169'):
        return build_densenet169(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('densenet201', 'dn201'):
        return build_densenet201(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('convnextt', 'convnext_t', 'convnexttiny', 'convnext-tiny'):
        return build_convnext_tiny(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('convnexts', 'convnext_s', 'convnextsmall', 'convnext-small'):
        return build_convnext_small(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('convnextb', 'convnext_base', 'convnextbase', 'convnext-b'):
        return build_convnext_base(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('convnextl', 'convnextlarge', 'convnext_large', 'convnext-l'):
        return build_convnext_large(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('mobilenet', 'mobilenetv1', 'mnv1'):
        return build_mobilenetv1(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('mobilenetv2', 'mnv2'):
        return build_mobilenetv2(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('mobilenetv3small', 'mnv3s'):
        return build_mobilenetv3small(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('mobilenetv3large', 'mnv3l'):
        return build_mobilenetv3large(input_shape=input_shape, train_base=train_base, lr=lr, weights=weights)
    if arch in ('resnet18', 'resnet34'):
        if _build_resnet18_small is None or _build_resnet34_small is None:
            raise ImportError('models.resnet_small module not available')
        if arch == 'resnet18':
            model = _build_resnet18_small(input_shape=input_shape, num_outputs=1)
        else:
            model = _build_resnet34_small(input_shape=input_shape, num_outputs=1)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        return model
    if arch in ('custom_cnn', 'cnn'):
        return build_custom_cnn(input_shape=input_shape, lr=lr)
    raise ValueError(f'Unknown architecture: {arch}')
