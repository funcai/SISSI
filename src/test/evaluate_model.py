from test_model import TestModel
import pandas as pd
import os.path as path
from itertools import combinations
import numpy as np

# TODO
# add following evaluation techniques
# - mse
# - accuracy
# - distances in class (norm somehow)
# - distances to others that are similar / not similar (norm somehow)
# - sparse-crossentropy loss
# visualization
# - PCA diagramm for classes (maybe create new file with Class only for that task)

class EvaluateModel:
    def __init__(self):
        self.tester = TestModel()

        test_data_path = path.abspath(path.join(__file__ ,"../../..")) + "/data/" + "wiki_de_test.csv"
        self.test_data = pd.read_csv(test_data_path)
        #self.test_data = self.test_data[5]
        print(len(self.test_data))

        # predictions = self.tester.predict(["hello","oh"])
        # print(len(predictions))

    # def get_mse(self):

    def get_distances_in_group(self):
        distances = {}
        for label in self.test_data.label.unique()[:2]:
            print("Calculating distances for label {}".format(label))
            label_dist = []
            label_data = self.test_data[self.test_data.label == label]
            for a, b in combinations(label_data.text, 2):
                pred_a = self.tester.predict(a)
                pred_b = self.tester.predict(b)
                dist = np.linalg.norm(pred_a-pred_b)
                label_dist.append(dist)

            distances[label] = np.mean(label_dist)

        # TODO norm distances
        print("Distances in group: {}".format(distances))
                
    def get_distances_to_similar(self):
        # is it fine to use new text here?
        a = "Mein rotes Auto ist vor der Bäckerei kaputt gegangen" # 0
        b = "Ich fahre regelmäßig mit dem Auto zur Arbeit wenn das Wetter schlecht ist" # 1
        c = "Während der Woche backe ich brot, ich bin ein Bäcker" # 2
        d = "Regen und Sturm mach mich traurig" # 3
        test_sentences = [a,b,c,d]
        # should be close : 0-1, 0-2, 1-3
        # should not be close: 0-3, 1-2, 2-3
        distances = {}
        for x, y in combinations(enumerate(test_sentences), 2):
            pred_x = self.tester.predict(x[1])
            pred_y = self.tester.predict(y[1])
            dist = np.linalg.norm(pred_x - pred_y)
            __label = str(x[0]) + "-" + str(y[0])
            distances[__label] = dist
        print("Similarity distances: {}".format(distances))
        # print("Following should be small: {}, {}, {}")

    def search_engine_test(self):
        search = "Bekannte Kurzfilme"
        # +1 for good result, -1 for wrong result, +2 for not obvious result
        score = 1
        results = [["Der Film Pulp Fiction ist einer der besten die jemals produziert wurden", score],
                    ["Kräfte sind in der Physik sehr relevant", -score],
                    ["Arne dreht gerne kurze Videos, er ist einer der bekanntesten", score*2],
                    ["Brot wird aus Weizen gebacken", -score]]
        
        search_pred = self.tester.predict(search)
        result_pred = [self.tester.predict(x[0]) for x in results]

        # which results has which ranking?
        # k nearest neighboors?

if __name__=="__main__":
    evaluate = EvaluateModel()
    evaluate.get_distances_in_group()
    evaluate.get_distances_to_similar()