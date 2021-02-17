import os.path as path
import sys
import csv
import time
sys.path.append(path.abspath(path.join(__file__ ,"../../preprocessing/vectorization/")))
from SpacyVectorizer import SpacyVectorizer
from RobertaVectorizer import RobertaVectorizer

samples = []
with open(path.abspath(path.join(__file__ ,"../../../data/wikipedia.csv"))) as csvfile:
  wikireader = csv.reader(csvfile)
  c = 0
  for row in wikireader:
    c = c + 1
    if(c < 100):
      # Use the first 512 characters of the text, because roberta can't handle more
      samples.append(row[0][0:512])


spacy = SpacyVectorizer()
start = time.time()
for sample in samples:
  spacy.vectorize(sample)
end = time.time()
print("Spacy took {:.3f}s total, {:.4f}s per sample ".format(end - start, (end - start) / len(samples)))


roberta = RobertaVectorizer()
start = time.time()
for sample in samples:
  roberta.vectorize(sample)
end = time.time()
print("Roberta took {:.3f}s total, {:.4f}s per sample ".format(end - start, (end - start) / len(samples)))
