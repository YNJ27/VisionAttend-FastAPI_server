#verification fn
import tensorflow as tf
import os
import numpy as np
from preprocessing import preprocess
from distance_model import distanceModel

def verify(detection_threshold, verification_threshold):
    # Detection Threshold: Metric above which a prediciton is considered positive 
    # Verification Threshold: Proportion of positive predictions / total positive samples
    
    Final = []
    resultsy = []
    resultsc = []
    resultsa = []
    resultsj = []

    #Verification with yug
    for image in os.listdir(os.path.join('Real_time_detection', 'Yug')):
        input_img = preprocess(os.path.join('Real_time_detection', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('Real_time_detection', 'Yug', image))

        input_img = tf.expand_dims(input_img, axis = 0)
        validation_img = tf.expand_dims(validation_img, axis = 0)
        
        # Make Predictions 
        result = distanceModel(input_img, validation_img)
        resultsy.append(result)
        
    detectiony = np.sum(np.array(resultsy) < detection_threshold)
    verificationy = detectiony / len(os.listdir(os.path.join('Real_time_detection', 'Yug'))) 
    Final.append(verificationy)

    
    #Verification with Chinmay
    for image in os.listdir(os.path.join('Real_time_detection', 'Chinmay')):
        input_img = preprocess(os.path.join('Real_time_detection', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('Real_time_detection', 'Chinmay', image))

        input_img = tf.expand_dims(input_img, axis = 0)
        validation_img = tf.expand_dims(validation_img, axis = 0)
        
        # Make Predictions 
        result = distanceModel(input_img, validation_img)
        resultsc.append(result)

    detectionc = np.sum(np.array(resultsc) < detection_threshold)
    verificationc = detectionc / len(os.listdir(os.path.join('Real_time_detection', 'Chinmay'))) 
    Final.append(verificationc)

    
    #Verification with Arjun
    for image in os.listdir(os.path.join('Real_time_detection', 'Arjun')):
        input_img = preprocess(os.path.join('Real_time_detection', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('Real_time_detection', 'Arjun', image))

        input_img = tf.expand_dims(input_img, axis = 0)
        validation_img = tf.expand_dims(validation_img, axis = 0)
        
        # Make Predictions 
        result = distanceModel(input_img, validation_img)
        resultsa.append(result)

    detectiona = np.sum(np.array(resultsa) < detection_threshold)
    verificationa = detectiona / len(os.listdir(os.path.join('Real_time_detection', 'Arjun'))) 
    Final.append(verificationa)
    
    
    #Verification with Jay
    for image in os.listdir(os.path.join('Real_time_detection', 'Jay')):
        input_img = preprocess(os.path.join('Real_time_detection', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('Real_time_detection', 'Jay', image))

        input_img = tf.expand_dims(input_img, axis = 0)
        validation_img = tf.expand_dims(validation_img, axis = 0)
        
        # Make Predictions 
        result = distanceModel(input_img, validation_img)
        resultsj.append(result)

    detectionj = np.sum(np.array(resultsj) < detection_threshold)
    verificationj = detectionj / len(os.listdir(os.path.join('Real_time_detection', 'Jay'))) 
    Final.append(verificationj)


    ###Finding the one with max value of verification and checking whether it crosses the verification threshold or not###
    students = ["Yug", "Chinmay","Arjun", "Jay"]
    
    # pos = 0
    # max = 0
    # for i in range(4):
    #     if(Final[i] > max):
    #         max = Final[i]
    #         pos = i

    #####The above logic can easily be implemented using np.argmax() and np.max()#######
    index = np.argmax(Final)
    max = np.max(Final)

    person = "Unknown"
    verified = (max >= verification_threshold)

    if(verified == True):
        person = students[index]
    
    return person, Final