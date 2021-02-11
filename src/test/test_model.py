# add interpreter

import sys
from tensorflow.keras.models import load_model
from tensorflow import constant as tf_constant
import os.path as path

import sys
sys.path.insert(1, path.abspath(path.join(__file__ ,"../../preprocessing/")))
from preprocessor import Preprocessor


class TestModel:
    def __init__(self, model_name="test.hdf5"):
        print('Starting test of {}'.format(model_name))
        models_path = path.abspath(path.join(__file__ ,"../../..")) + "/models/" + model_name
        print(models_path)
        self.model = load_model(models_path)
        self.preprocessor = Preprocessor()

    def predict(self, text):
        word_vectors = self.preprocessor.vectorize(text)
        predicted_vectors = self.model(tf_constant([word_vectors]), training=False)
        return predicted_vectors

if __name__ == "__main__":
    # get input model name
    model_name = sys.argv[1] if len(sys.argv)>1 else "test.hdf5"
    my_model = TestModel(model_name)
    print(my_model.predict("hello"))