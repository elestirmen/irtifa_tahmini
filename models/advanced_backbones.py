import tensorflow as tf
from tensorflow.keras import Model, layers


def _ensure_rgb(input_shape):
    if input_shape[-1] != 3:
        raise ValueError('Bu mimari yalnızca 3 kanallı (RGB) girişleri destekler.')


def _make_divisible(val, divisor=8):
    return int((val + divisor - 1) // divisor * divisor)


def build_squeezenet_backbone(input_shape):
    """SqueezeNet 1.1 backbone without classification head."""
    _ensure_rgb(input_shape)

    def fire_module(x, squeeze_filters, expand_filters, name):
        with tf.name_scope(name):
            squeezed = layers.Conv2D(squeeze_filters, (1, 1), activation='relu', padding='same', name=f'{name}_squeeze')(x)
            expand_1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu', padding='same', name=f'{name}_expand_1x1')(squeezed)
            expand_3x3 = layers.Conv2D(expand_filters, (3, 3), activation='relu', padding='same', name=f'{name}_expand_3x3')(squeezed)
            return layers.Concatenate(name=f'{name}_concat')([expand_1x1, expand_3x3])

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', name='sq_conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='sq_pool1')(x)

    x = fire_module(x, 16, 64, name='fire2')
    x = fire_module(x, 16, 64, name='fire3')
    x = fire_module(x, 32, 128, name='fire4')
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='sq_pool2')(x)

    x = fire_module(x, 32, 128, name='fire5')
    x = fire_module(x, 48, 192, name='fire6')
    x = fire_module(x, 48, 192, name='fire7')
    x = fire_module(x, 64, 256, name='fire8')
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='sq_pool3')(x)

    x = fire_module(x, 64, 256, name='fire9')
    x = layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='sq_conv10')(x)
    x = layers.BatchNormalization(name='sq_bn')(x)

    return Model(inputs, x, name='squeezenet_backbone')


class ClassToken(layers.Layer):
    def build(self, input_shape):
        dim = input_shape[-1]
        self.cls = self.add_weight('cls', shape=(1, 1, dim), initializer='zeros')

    def call(self, x):
        batch = tf.shape(x)[0]
        cls = tf.broadcast_to(self.cls, [batch, 1, tf.shape(x)[-1]])
        return tf.concat([cls, x], axis=1)


class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_positions, dim, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.dim = dim

    def build(self, input_shape):
        self.pos = self.add_weight('pos', shape=(1, self.num_positions, self.dim), initializer='zeros')

    def call(self, x):
        return x + self.pos


def _vit_transformer_block(x, dim, num_heads, mlp_dim, dropout, name):
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln1')(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout, name=f'{name}_attn')(x, x)
    x = layers.Dropout(dropout, name=f'{name}_attn_drop')(x)
    x = layers.Add(name=f'{name}_attn_add')([shortcut, x])

    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(x)
    x = layers.Dense(mlp_dim, activation='gelu', name=f'{name}_mlp_dense1')(x)
    x = layers.Dropout(dropout, name=f'{name}_mlp_drop1')(x)
    x = layers.Dense(dim, name=f'{name}_mlp_dense2')(x)
    x = layers.Dropout(dropout, name=f'{name}_mlp_drop2')(x)
    return layers.Add(name=f'{name}_mlp_add')([shortcut, x])


VIT_CONFIGS = {
    'vit_tiny': dict(embed_dim=192, depth=12, num_heads=3, mlp_dim=768, patch_size=16),
    'vit_small': dict(embed_dim=384, depth=12, num_heads=6, mlp_dim=1536, patch_size=16),
    'vit_base': dict(embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, patch_size=16),
    'vit_large': dict(embed_dim=1024, depth=24, num_heads=16, mlp_dim=4096, patch_size=16),
}


def build_vit_backbone(variant, input_shape, dropout=0.0):
    _ensure_rgb(input_shape)
    if variant not in VIT_CONFIGS:
        raise ValueError(f'Bilinmeyen ViT varyantı: {variant}')
    cfg = VIT_CONFIGS[variant]
    h, w = input_shape[:2]
    if h % cfg['patch_size'] != 0 or w % cfg['patch_size'] != 0:
        raise ValueError('Input boyutu patch_size ile bölünebilir olmalıdır.')
    num_patches = (h // cfg['patch_size']) * (w // cfg['patch_size'])

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(cfg['embed_dim'], kernel_size=cfg['patch_size'], strides=cfg['patch_size'], padding='valid', name='vit_patch_embed')(inputs)
    x = layers.Reshape((num_patches, cfg['embed_dim']), name='vit_flatten')(x)
    x = ClassToken(name='vit_cls_token')(x)
    x = AddPositionEmbedding(num_positions=num_patches + 1, dim=cfg['embed_dim'], name='vit_pos')(x)

    for idx in range(cfg['depth']):
        x = _vit_transformer_block(x, cfg['embed_dim'], cfg['num_heads'], cfg['mlp_dim'], dropout, name=f'vit_block_{idx}')

    x = layers.LayerNormalization(epsilon=1e-6, name='vit_ln')(x)
    cls_token = layers.Lambda(lambda t: t[:, 0], name='vit_cls_extract')(x)
    return Model(inputs, cls_token, name=f'{variant}_backbone')


class SpatialReductionAttention(layers.Layer):
    def __init__(self, dim, num_heads, sr_ratio=1, dropout=0.0, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.drop = layers.Dropout(dropout) if dropout else None
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=dropout)
        if sr_ratio > 1:
            self.sr_conv = layers.Conv2D(dim, kernel_size=sr_ratio, strides=sr_ratio, padding='same')
            self.sr_norm = layers.LayerNormalization(epsilon=1e-5)
        else:
            self.sr_conv = None
        self.proj = layers.Dense(dim)

    def call(self, x, training=None):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        tokens = h * w
        seq = tf.reshape(x, [b, tokens, self.dim])
        if self.sr_conv is not None:
            sr = self.sr_conv(x)
            sr = self.sr_norm(sr)
            sr_tokens = tf.shape(sr)[1] * tf.shape(sr)[2]
            kv = tf.reshape(sr, [b, sr_tokens, self.dim])
        else:
            kv = seq
        attn_out = self.attn(query=seq, value=kv, key=kv, training=training)
        attn_out = self.proj(attn_out)
        attn_out = tf.reshape(attn_out, [b, h, w, self.dim])
        if self.drop:
            attn_out = self.drop(attn_out, training=training)
        return attn_out


class DepthwiseConvMixer(layers.Layer):
    def __init__(self, dim, kernel_size=3, dropout=0.0, name=None):
        super().__init__(name=name)
        self.dw = layers.DepthwiseConv2D(kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')
        self.pw = layers.Conv2D(dim, 1)
        self.drop = layers.Dropout(dropout) if dropout else None

    def call(self, x, training=None):
        x = self.dw(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.pw(x)
        if self.drop:
            x = self.drop(x, training=training)
        return x


class MLPBlock(layers.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0, use_dwconv=False, name=None):
        super().__init__(name=name)
        hidden_dim = int(dim * mlp_ratio)
        self.use_dwconv = use_dwconv
        self.fc1 = layers.Conv2D(hidden_dim, 1)
        self.dw = layers.DepthwiseConv2D(3, padding='same') if use_dwconv else None
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Conv2D(dim, 1)
        self.drop = layers.Dropout(dropout) if dropout else None

    def call(self, x, training=None):
        x = self.fc1(x)
        if self.dw is not None:
            x = self.dw(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.drop:
            x = self.drop(x, training=training)
        return x


class MetaFormerBlock(layers.Layer):
    def __init__(self, dim, token_mixer='attention', num_heads=4, sr_ratio=1, mlp_ratio=4.0, dropout=0.0, use_dwconv_mlp=False, name=None):
        super().__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        if token_mixer == 'sr_attention':
            self.mixer = SpatialReductionAttention(dim, num_heads, sr_ratio, dropout)
        elif token_mixer == 'attention':
            self.mixer = SpatialReductionAttention(dim, num_heads, sr_ratio=1, dropout=dropout)
        elif token_mixer == 'dwconv':
            self.mixer = DepthwiseConvMixer(dim, dropout=dropout)
        else:
            raise ValueError(f'Unsupported token mixer: {token_mixer}')
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLPBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout, use_dwconv=use_dwconv_mlp)
        self.drop = layers.Dropout(dropout) if dropout else None

    def call(self, x, training=None):
        shortcut = x
        x = self.norm1(x)
        x = self.mixer(x, training=training)
        if self.drop:
            x = self.drop(x, training=training)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x, training=training)
        if self.drop:
            x = self.drop(x, training=training)
        return shortcut + x


PVT_CONFIGS = {
    'pvt_tiny': dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvt_small': dict(embed_dims=[64, 128, 320, 512], depths=[2, 3, 4, 3], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvt_medium': dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvt_large': dict(embed_dims=[64, 128, 320, 512], depths=[3, 5, 8, 3], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
}


def build_pvt_backbone(variant, input_shape, dropout=0.0):
    _ensure_rgb(input_shape)
    if variant not in PVT_CONFIGS:
        raise ValueError(f'Bilinmeyen PVT varyantı: {variant}')
    cfg = PVT_CONFIGS[variant]
    inputs = layers.Input(shape=input_shape)
    x = inputs
    stages = len(cfg['embed_dims'])
    for stage in range(stages):
        stride = 4 if stage == 0 else 2
        x = layers.Conv2D(cfg['embed_dims'][stage], kernel_size=stride, strides=stride, padding='same', name=f'{variant}_patch_embed_{stage}')(x)
        x = layers.LayerNormalization(epsilon=1e-6, name=f'{variant}_ln_{stage}')(x)
        for block in range(cfg['depths'][stage]):
            x = MetaFormerBlock(
                dim=cfg['embed_dims'][stage],
                token_mixer='sr_attention',
                num_heads=cfg['heads'][stage],
                sr_ratio=cfg['sr'][stage],
                mlp_ratio=4.0,
                dropout=dropout,
                name=f'{variant}_stage{stage}_block{block}',
            )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{variant}_out_ln')(x)
    x = layers.GlobalAveragePooling2D(name=f'{variant}_gap')(x)
    return Model(inputs, x, name=f'{variant}_backbone')


PVT_V2_CONFIGS = {
    'pvtv2_b0': dict(embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvtv2_b1': dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvtv2_b2': dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvtv2_b3': dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvtv2_b4': dict(embed_dims=[64, 128, 320, 512], depths=[3, 8, 27, 3], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
    'pvtv2_b5': dict(embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3], heads=[1, 2, 5, 8], sr=[8, 4, 2, 1]),
}


def _pvtv2_patch_embed(x, out_dim, stride, name):
    x = layers.Conv2D(out_dim, kernel_size=stride, strides=stride, padding='same', name=f'{name}_conv')(x)
    x = layers.DepthwiseConv2D(3, padding='same', name=f'{name}_dw')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation('relu', name=f'{name}_act')(x)
    return x


def build_pvtv2_backbone(variant, input_shape, dropout=0.0):
    _ensure_rgb(input_shape)
    if variant not in PVT_V2_CONFIGS:
        raise ValueError(f'Bilinmeyen PVT-v2 varyantı: {variant}')
    cfg = PVT_V2_CONFIGS[variant]
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for stage in range(len(cfg['embed_dims'])):
        stride = 4 if stage == 0 else 2
        x = _pvtv2_patch_embed(x, cfg['embed_dims'][stage], stride=stride, name=f'{variant}_patch_{stage}')
        for block in range(cfg['depths'][stage]):
            x = MetaFormerBlock(
                dim=cfg['embed_dims'][stage],
                token_mixer='sr_attention',
                num_heads=cfg['heads'][stage],
                sr_ratio=cfg['sr'][stage],
                mlp_ratio=4.0,
                dropout=dropout,
                use_dwconv_mlp=True,
                name=f'{variant}_stage{stage}_block{block}',
            )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{variant}_out_ln')(x)
    x = layers.GlobalAveragePooling2D(name=f'{variant}_gap')(x)
    return Model(inputs, x, name=f'{variant}_backbone')


EFFICIENTFORMER_CONFIGS = {
    'efficientformer_l1': dict(widths=[48, 96, 224, 448], depths=[3, 2, 6, 4], attn_blocks=[1, 1, 1, 1]),
    'efficientformer_l3': dict(widths=[64, 128, 320, 512], depths=[4, 4, 12, 6], attn_blocks=[1, 1, 3, 2]),
    'efficientformer_l7': dict(widths=[96, 192, 384, 768], depths=[6, 6, 18, 8], attn_blocks=[2, 2, 6, 4]),
}


def build_efficientformer_backbone(variant, input_shape, dropout=0.0):
    _ensure_rgb(input_shape)
    if variant not in EFFICIENTFORMER_CONFIGS:
        raise ValueError(f'Bilinmeyen EfficientFormer varyantı: {variant}')
    cfg = EFFICIENTFORMER_CONFIGS[variant]
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(cfg['widths'][0], 3, strides=2, padding='same', name=f'{variant}_stem_conv1')(inputs)
    x = layers.BatchNormalization(name=f'{variant}_stem_bn1')(x)
    x = layers.Activation('relu', name=f'{variant}_stem_act1')(x)
    x = layers.Conv2D(cfg['widths'][0], 3, strides=2, padding='same', name=f'{variant}_stem_conv2')(x)
    x = layers.BatchNormalization(name=f'{variant}_stem_bn2')(x)
    x = layers.Activation('relu', name=f'{variant}_stem_act2')(x)

    for stage, (width, depth, attn_count) in enumerate(zip(cfg['widths'], cfg['depths'], cfg['attn_blocks'])):
        for block in range(depth):
            use_attn = block < attn_count
            token_mixer = 'attention' if use_attn else 'dwconv'
            x = MetaFormerBlock(
                dim=width,
                token_mixer=token_mixer,
                num_heads=max(width // 32, 1),
                sr_ratio=1,
                mlp_ratio=4.0,
                dropout=dropout,
                name=f'{variant}_stage{stage}_block{block}',
            )(x)
        if stage < len(cfg['widths']) - 1:
            next_width = cfg['widths'][stage + 1]
            x = layers.Conv2D(next_width, 3, strides=2, padding='same', name=f'{variant}_down_{stage}')(x)
            x = layers.BatchNormalization(name=f'{variant}_down_{stage}_bn')(x)
            x = layers.Activation('relu', name=f'{variant}_down_{stage}_act')(x)

    x = layers.LayerNormalization(epsilon=1e-6, name=f'{variant}_out_ln')(x)
    x = layers.GlobalAveragePooling2D(name=f'{variant}_gap')(x)
    return Model(inputs, x, name=f'{variant}_backbone')


EFFICIENTVIT_CONFIGS = {
    'efficientvit_m0': dict(widths=[32, 64, 128, 256], depths=[2, 3, 4, 2], attn_stages=[False, True, True, True]),
    'efficientvit_m1': dict(widths=[48, 96, 192, 320], depths=[2, 3, 6, 2], attn_stages=[False, True, True, True]),
    'efficientvit_m2': dict(widths=[64, 128, 256, 384], depths=[2, 4, 8, 3], attn_stages=[False, True, True, True]),
    'efficientvit_m3': dict(widths=[80, 160, 320, 448], depths=[2, 4, 10, 3], attn_stages=[False, True, True, True]),
    'efficientvit_m4': dict(widths=[96, 192, 384, 512], depths=[3, 4, 12, 3], attn_stages=[False, True, True, True]),
    'efficientvit_m5': dict(widths=[112, 224, 448, 640], depths=[3, 4, 16, 4], attn_stages=[False, True, True, True]),
}


def build_efficientvit_backbone(variant, input_shape, dropout=0.0):
    _ensure_rgb(input_shape)
    if variant not in EFFICIENTVIT_CONFIGS:
        raise ValueError(f'Bilinmeyen EfficientViT varyantı: {variant}')
    cfg = EFFICIENTVIT_CONFIGS[variant]
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(cfg['widths'][0], 3, strides=2, padding='same', name=f'{variant}_stem_conv')(inputs)
    x = layers.BatchNormalization(name=f'{variant}_stem_bn')(x)
    x = layers.Activation('relu', name=f'{variant}_stem_act')(x)

    for stage, (width, depth, use_attn) in enumerate(zip(cfg['widths'], cfg['depths'], cfg['attn_stages'])):
        for block in range(depth):
            token_mixer = 'sr_attention' if use_attn else 'dwconv'
            sr_ratio = 2 if use_attn else 1
            x = MetaFormerBlock(
                dim=width,
                token_mixer=token_mixer,
                num_heads=max(width // 64, 1),
                sr_ratio=sr_ratio,
                mlp_ratio=3.0,
                dropout=dropout,
                use_dwconv_mlp=True,
                name=f'{variant}_stage{stage}_block{block}',
            )(x)
        if stage < len(cfg['widths']) - 1:
            next_width = cfg['widths'][stage + 1]
            x = layers.Conv2D(next_width, 3, strides=2, padding='same', name=f'{variant}_down_{stage}')(x)
            x = layers.BatchNormalization(name=f'{variant}_down_{stage}_bn')(x)
            x = layers.Activation('relu', name=f'{variant}_down_{stage}_act')(x)

    x = layers.LayerNormalization(epsilon=1e-6, name=f'{variant}_out_ln')(x)
    x = layers.GlobalAveragePooling2D(name=f'{variant}_gap')(x)
    return Model(inputs, x, name=f'{variant}_backbone')


__all__ = [
    'build_squeezenet_backbone',
    'build_vit_backbone',
    'build_pvt_backbone',
    'build_pvtv2_backbone',
    'build_efficientformer_backbone',
    'build_efficientvit_backbone',
]
