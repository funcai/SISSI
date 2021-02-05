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

raw_data = pd.read_csv("../../data/wikipedia.csv")
# clean out where text is NaN
raw_data = raw_data[raw_data.text.notna()]

x = [train_preprocessor.vectorize(sentence) for sentence in raw_data.text]
y = raw_data.label.to_numpy()

features = tf.constant(x, shape=(1,len(x),300))
labels = tf.constant(y, shape=(1,len(y),1))

features_dataset = tf.data.Dataset.from_tensor_slices(features)
labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

train_dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

model = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation='relu', input_shape=(300,)),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(300)
])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

train_loss_results = []
train_accuracy_results = []
num_epochs = 5000
import time
for epoch in range(num_epochs):
  start = time.time()
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(x, training=True)
    #new_y = get_new_y(predictions, y)
    epoch_accuracy.update_state(y, predictions)

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  end = time.time()
  print("Epoch {:03d}: Loss: {:.3f}, MAE: {:.3}, time: {:.8f}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result(), end - start))

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