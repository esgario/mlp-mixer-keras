import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Dense, Layer, Input, LayerNormalization, Permute
)
from einops.layers.tensorflow import Rearrange, Reduce


class MlpBlock(Layer):
    def __init__(self, dim, hidden_dim, **kwargs):
        super(MlpBlock, self).__init__(**kwargs)
        self.dim = dim
        self.dense1 = Dense(hidden_dim, activation=tf.nn.gelu)
        self.dense2 = Dense(dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)


class MixerBlock(Layer):
    def __init__(
            self,
            num_patches,
            channel_dim,
            tokens_mlp_dim,
            channels_mlp_dim,
            **kwargs
    ):
        super(MixerBlock, self).__init__(**kwargs)
        self.layer_norm = [LayerNormalization(axis=1) for _ in range(2)]
        self.permute = Permute((2, 1))
        self.mlp_block = [
            MlpBlock(num_patches, tokens_mlp_dim, name='token_mixing'),
            MlpBlock(channel_dim, channels_mlp_dim, name='channel_mixing')
        ]

    def call(self, x):
        y = self.layer_norm[0](x)
        y = self.permute(y)
        y = self.mlp_block[0](y)
        y = self.permute(y)
        x = x + y
        y = self.layer_norm[1](x)
        return x + self.mlp_block[1](y)

    def compute_output_shape(self, input_shape):
        return input_shape


def _mlp_mixer(
    input_shape=(224, 224, 3),
    n_classes=10,
    n_blocks=8,
    patch_size=16,
    hidden_dim=512,
    tokens_mlp_dim=256,
    channels_mlp_dim=2048
):
    height, width, _ = input_shape
    num_patches = (height*width)//(patch_size**2)
    inputs = Input(shape=input_shape)

    x = Conv2D(
        hidden_dim,
        kernel_size=patch_size,
        strides=patch_size,
        name='stem')(inputs)

    x = Rearrange('n h w c -> n (h w) c')(x)

    for _ in range(n_blocks):
        x = MixerBlock(
            num_patches=num_patches,
            channel_dim=hidden_dim,
            tokens_mlp_dim=tokens_mlp_dim,
            channels_mlp_dim=channels_mlp_dim
        )(x)

    x = LayerNormalization(name='pre_head_layer_norm')(x)
    x = Reduce('b n c -> b c', 'mean')(x)
    out = Dense(n_classes, name='head')(x)

    return Model(inputs, out)


def mlp_mixer(spec_code, input_shape=(224, 224, 3), n_classes=10):
    """Create an MLP Mixer model.

    Parameters
    ----------
    spec_code: str
        Specification code according to the parameters presented in
        table 1 of the paper. It must assume one of theses values:
        `[ 'S32', 'S16', 'B32', 'B16', 'L32', 'L16', 'H14' ]`
    input_shape: tuple
        Input image shape.
    n_classes: int
        Maximum number of classes that exists in the dataset.

    Return
    ------
        Returns the selected MLP Mixer model.

    Reference
    ---------
    MLP-Mixer: An all-MLP Architecture for Vision
    https://arxiv.org/pdf/2105.01601.pdf
    """
    PARAMETERS = {
        "S32": [8, 32, 512, 256, 2048],
        "S16": [8, 16, 512, 256, 2048],
        "B32": [12, 32, 768, 384, 3072],
        "B16": [12, 16, 768, 384, 3072],
        "L32": [24, 32, 1024, 512, 4096],
        "L16": [24, 16, 1024, 512, 4096],
        "H14": [32, 14, 1280, 640, 5120]
    }

    return _mlp_mixer(input_shape, n_classes, *PARAMETERS[spec_code])
