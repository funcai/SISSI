from VectorizerInterface import VectorizerInterface, Vectors, OOVWords
import spacy
import numpy as np

class SpacyVectorizer(VectorizerInterface):
    def __init__(self):
        self.nlp = spacy.load('de_core_news_lg')

    def vectorize(self, text: str) -> Vectors:
        clean_text = ''.join([word.lemma_ + " " for word in self.nlp(text) if not word.is_punct and not word.is_stop and not word.is_oov and not word.like_num and not word.like_url]).strip()
        return np.asarray(self.nlp(clean_text).vector).astype('float32')

    def getOOVWords(self, text: str) -> OOVWords:
      return [word for word in self.nlp(text) if word.is_oov]