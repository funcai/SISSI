import spacy
import numpy as np
class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('de_core_news_lg')

    def normalize(self, text):
        doc = self.nlp(text)
        lemma_text = ''.join([token.lemma_ + token.whitespace_ for token in doc]).strip()
        lemma_doc = self.nlp(lemma_text)
        lower_text = ''.join([token.lower_ + token.whitespace_ for token in lemma_doc]).strip()
        return self.nlp(lower_text)

    def vectorize(self, text):
        norm_doc = self.normalize(text)

        with self.nlp.disable_pipes():
            return norm_doc.vector

    def normalize_words(self,text):
        words = ''.join([word.lemma_ + " " for word in self.nlp(text) if not word.is_punct and not word.is_stop and not word.is_oov and not word.like_num and not word.like_url]).strip()
        return np.asarray(self.nlp(words))
        #vectors = [word.vector for word in self.nlp(words)]
        #return np.asarray(vectors).astype('float32')
    
    def text_vector(self,text):
        vectors = self.normalize_words(self, text)
        return np.mean(vectors)