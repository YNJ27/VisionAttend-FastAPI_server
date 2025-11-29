from embeddingLayer import embedding_model
import numpy as np

def distanceModel(anc, pos):
    
    emebedding_anc = embedding_model(anc)
    emebedding_pos = embedding_model(pos)
    
    distance = np.sum(np.square(emebedding_anc - emebedding_pos), axis=-1)
    return distance