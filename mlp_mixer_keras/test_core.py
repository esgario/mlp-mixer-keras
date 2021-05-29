import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from .core import mlp_mixer


@pytest.fixture
def image_batch():
    np.random.seed = 1
    return np.random.random((3, 224, 224, 3))


@pytest.mark.parametrize(
    "spec_code, n_parameters",
    [("S32", 19), ("S16", 18), ("B32", 60), ("B16", 59),
     ("L32", 206), ("L16", 207), ("H14", 431)]
)
def test_mlp_mixer(image_batch, spec_code, n_parameters):
    input_shape = image_batch.shape[1:]
    n_classes = 100
    with tf.device('cpu'):
        model = mlp_mixer(spec_code, input_shape, n_classes)
        pred = model.predict(image_batch)
    assert pred.shape[0] == image_batch.shape[0]
    assert pred.shape[1] == n_classes

    trainable_count = round(np.sum(
        [K.count_params(w) for w in model.trainable_weights]
    ) / 1000000)

    assert trainable_count == n_parameters
