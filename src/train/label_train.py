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
train_preprocessor = Preprocessor()

raw_data = pd.read_csv("../../data/wiki_de.csv")
# clean out where text is NaN
raw_data = raw_data[raw_data.text.notna()]

x = [train_preprocessor.vectorize(sentence) for sentence in raw_data.text]
y = raw_data.label.to_numpy()

features = tf.constant(x, shape=(1,len(x),300))
labels = tf.constant(y, shape=(1,len(y),1))

features_dataset = tf.data.Dataset.from_tensor_slices(features)
labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

split = 9
train_dataset = dataset.window(split, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds))
validation_dataset = dataset.skip(split).window(1, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds))

model = tf.keras.Sequential([
  tf.keras.layers.Dense(1000, activation='relu', input_shape=(300,)),
  tf.keras.layers.Dense(1000, activation='relu'),
  tf.keras.layers.Dense(1000, activation='relu'),
  tf.keras.layers.Dense(300)
])

num_epochs = 200
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_object, optimizer="adam", metrics=["accuracy"])


model.fit(train_dataset, validation_data=validation_dataset, epochs=num_epochs)
#model.evaluate(validation_dataset)
model.save('../../models/test.hdf5')

def test(text):
    word_vectors = train_preprocessor.text_vector(text)
    predicted_vectors = model(tf.constant([word_vectors]), training=False)
    return predicted_vectors
    return np.average(predicted_vectors, axis=0)

a = "Brot (ahd. prôt, von urgerm. *brauda-) ist ein traditionelles Nahrungsmittel, das aus einem Teig aus gemahlenem Getreide (Mehl), Wasser, einem Triebmittel und meist weiteren Zutaten gebacken wird. Brot zählt zu den Grundnahrungsmitteln."
b = "Eine enge Gangabstufung ist auch für Transportarbeiten günstig, da das Verhältnis von Leistung und Gesamtgewicht des Zuges bei Traktoren häufig geringer ist als bei Lastkraftwagen."
#a = "Weizen"
#b = "Luftmasche"
query = "Brot"
a = test(a)
b = test(b)
query = test(query)


diff1 = np.linalg.norm(query - a)
diff2 = np.linalg.norm(query - b)
diff3 = np.linalg.norm(a - b)

print(diff1)
print(diff2)
print(diff3)