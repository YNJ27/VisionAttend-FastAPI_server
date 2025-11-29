import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class DistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='distance_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred shape: (batch_size, 2)
        d_ap = y_pred[:, 0]  # anchor-positive
        d_an = y_pred[:, 1]  # anchor-negative

        correct = tf.cast(d_ap < d_an, tf.float32)
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.cast(tf.size(correct), tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
