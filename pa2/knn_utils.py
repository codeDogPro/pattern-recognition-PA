import utils
import pandas as pd


class KnnModel(object):
    def __init__(self):
        self.model_data = None
        self.model_label = None
        self.K = 5

    def train(self, train_data, K=5):
        label = train_data.shape[1]
        self.model_data, self.model_label = train_data[:, 0:label - 1], train_data[:label]
        self.K = K

    def train_fisher(self, train_data, K=5):
        self.K = K

    def validation(self, test):
        label = test.shape[1]
        test_data, test_label = test[:, 0:label - 1], test[:, label - 1:label]
        correct_cnt = 0
        for i in range(test.shape[0]):
            result = self.predict(test_data[i])
            if result == test_label[i]:
                correct_cnt += 1
        return correct_cnt / test.shape[0]

    def predict(self, data):
        self.model_data -= data
        return 0


def run_with_KFold(data, K_split, K_near, use_fisher, Log):
    model = KnnModel()
    accuracy = utils.Kfold_model(data, K_split, K_near, use_fisher, model, Log)
    print("average accuracy:", accuracy)
