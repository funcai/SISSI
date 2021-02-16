import os.path as path
import sys
import csv
import time
sys.path.append(path.abspath(path.join(__file__ ,"../../preprocessing/vectorization/")))
from SpacyVectorizer import SpacyVectorizer

samples = []
with open(path.abspath(path.join(__file__ ,"../../../data/wikipedia.csv"))) as csvfile:
  wikireader = csv.reader(csvfile)
  for row in wikireader:
    samples.append(row[0])


spacy = SpacyVectorizer()

start = time.time()
for sample in samples:
  spacy.vectorize(sample)
end = time.time()
print("Spacy took {:.3f}s total, {:.4f}s per sample ".format(end - start, (end - start) / len(samples)))