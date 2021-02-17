from VectorizerInterface import VectorizerInterface, Vectors, OOVWords
from transformers import AutoTokenizer

class RobertaVectorizer(VectorizerInterface):
    def __init__(self):
      self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-german")

    def vectorize(self, text: str) -> Vectors:
        return self.tokenizer(text)["input_ids"]

    def getOOVWords(self, text: str) -> OOVWords:
      return []