import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from preprocessor import Preprocessor

train_preprocessor = Preprocessor()

data = pd.read_csv("../../data/wikipedia.csv")
# clean out where text is NaN
data = data[data.text.notna()]

doc_vectors = np.array([train_preprocessor.vectorize(text) for text in data.text])

print(doc_vectors.shape)

# doc_vectores.shape => (359, 300)

X_train, X_test, y_train, y_test = train_test_split(doc_vectors, data.label, test_size=0.1, random_state=1)

# implement NN or boosting

# Set dual=False to speed up training, and it's not needed
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )
