import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, name="triplet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin

    def call(self,dummy_distance, distances):
    
        d_ap = distances[:,0]
        d_an = distances[:,1]
        margin = 0.5
    
        loss = tf.maximum(d_ap - d_an + self.margin, 0.0)
        return tf.reduce_mean(loss)