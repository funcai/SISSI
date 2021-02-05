import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from preprocessor import Preprocessor
import random
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import combinations

model = tf.keras.models.load_model('../../models/test.hdf5')

train_preprocessor = Preprocessor()

def test(text):
    word_vectors = train_preprocessor.vectorize(text)
    predicted_vectors = model(tf.constant([word_vectors]), training=False)
    return predicted_vectors
    return np.average(predicted_vectors, axis=0)

a = "Nahrungsmittel"
b = "Luftmasche"
#a = "Weizen"
#b = "Luftmasche"
query = "Brot"
a = test(a)
#b = test(b)
query = test(query)


diff1 = np.linalg.norm(query - a)
#diff2 = np.linalg.norm(query - b)
#diff3 = np.linalg.norm(b-a)

print(diff1)
#print(diff2)
#print(diff3)