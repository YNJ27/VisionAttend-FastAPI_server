import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def l2_norm(x):
    return tf.math.l2_normalize(x, axis=1)