from typing import List
Vectors = List[float]
OOVWords = List[str]

class VectorizerInterface:
    def vectorize(self, text: str) -> Vectors:
      # Vectorizes the given text into a list of floats
      return [0.0]
  
    def getOOVWords(self, text: str) -> OOVWords:
      # Returns a list of words which are out of dictionary
      return []