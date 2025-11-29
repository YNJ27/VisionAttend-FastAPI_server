import tensorflow as tf
from L2norm import l2_norm
from distanceLayer import DistanceLayer
from tripletLoss import TripletLoss
from accuracyMetric import DistanceMetric

model = tf.keras.models.load_model('siamesemodel3.keras', 
                                   custom_objects={'DistLayer':DistanceLayer, 'TripletLoss':TripletLoss, 'DistAcc':DistanceMetric, "l2_norm": l2_norm})

embedding_model = model.get_layer("Embedding")