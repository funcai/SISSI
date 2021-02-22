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
        # predictions = self.tester.predict(["hello","oh"])
        # print(len(predictions))
        self.predictions = None

    def predict_test_data(self):
        self.predictions = self.tester.predict(self.test_data.text)

    def get_mean_distances_in_group(self):
        mean_distances = {}
        median_distances = {}
        variances = {}

        for label in self.test_data.label.unique()[:2]:
            print("Calculating distances for label {}".format(label))
            label_dist = []
            label_data = self.test_data[self.test_data.label == label]
            for a, b in combinations(label_data.text, 2):
                pred = self.tester.predict([a, b])
                dist = np.linalg.norm(pred[0].numpy() - pred[1].numpy())
                # print(dist)
                label_dist.append(dist)

            mean_distances[label] = np.mean(label_dist)
            median_distances[label] = np.median(label_dist)
            variances[label] = np.var(label_dist)

        # TODO norm distances
        print("Mean distances in group: {}".format(mean_distances))
        print("Median distances in group: {}".format(median_distances))           

    def get_cosine_similarities(self):
        cos_similarities = {}

        if self.predictions == None:
            self.predict_test_data()

        for a, b in combinations(enumerate(self.predictions), 2):
            label_a = self.test_data.label[a[0]]
            label_b = self.test_data.label[b[0]]
            comb_label = str(label_a)+ "-" + str(label_b)

            cos_sim = np.dot(a[1], b[1]) / (np.linalg.norm(a[1]) * np.linalg.norm(b[1]))
            
            if not comb_label in cos_similarities:
                cos_similarities[comb_label] = [cos_sim]
            else:
                cos_similarities[comb_label].append(cos_sim)

        # print("len is: {}".format(cos_similarities))
        return cos_similarities

    def get_distances(self):
        distances = {}

        if self.predictions == None:
            self.predict_test_data()

        for a, b in combinations(enumerate(self.predictions), 2):
            label_a = self.test_data.label[a[0]]
            label_b = self.test_data.label[b[0]]
            comb_label = str(label_a)+ "-" + str(label_b)

            dist = np.linalg.norm(a[1] - b[1])
            
            if not comb_label in distances:
                distances[comb_label] = [dist]
            else:
                distances[comb_label].append(dist)

        # print("len is: {}".format(distances))
        return distances

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
        search =  ["Bekannte Kurzfilme"]
        # +1 for good result, -1 for wrong result, +2 for not obvious result
        # score = 1
        results = ["Der Film Pulp Fiction ist einer der besten die jemals produziert wurden",
                    "Kräfte sind in der Physik sehr relevant",
                    "Arne dreht gerne kurze Videos, er ist einer der bekanntesten",
                    "Brot wird aus Weizen gebacken"]
        
        search_pred = self.tester.predict(search)
        result_preds = self.tester.predict(results)
        distances = {}
        for i, result_pred in enumerate(result_preds):
            distances[i] = np.linalg.norm(search_pred[0] - result_pred)

        # order and print 
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        print(sorted_distances)
        # k nearest neighboors?

        # calculate score
        # should be [2,0,1,3] or [2,0,3,1]
        

if __name__=="__main__":
    evaluate = EvaluateModel()
    # evaluate.get_distances_in_group()
    # evaluate.get_distances_to_similar()
    #evaluate.search_engine_test()
    evaluate.predict_test_data()
    evaluate.get_cosine_similarities()
    print('Done')