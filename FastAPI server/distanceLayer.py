import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer

@register_keras_serializable()
class DistanceLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return tf.stack([ap_distance, an_distance], axis=1)
