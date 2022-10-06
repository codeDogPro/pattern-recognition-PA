import utils
import numpy as np
from collections import Counter


class KnnModel(object):
    def __init__(self):
        self.model_data = None
        self.model_label = None
        self.K = 5

    def train(self, train_data, K=5):
        label = train_data.shape[1]
        self.model_data, self.model_label = train_data[:, :label-1], train_data[:, label-1:]
        self.model_data = np.matrix(self.model_data, dtype=np.float64)
        self.K = K

    def train_fisher(self, train_data, K=5):
        self.K = K

    def validation(self, test):
        label = test.shape[1]
        test_data, test_label = test[:, :label-1], test[:, label-1:]
        test_data = np.matrix(test_data, dtype=np.float64)
        correct_cnt = 0
        for i in range(test.shape[0]):
            result = self.predict(test_data[i])
            print('label:', test_label[i])
            if result == test_label[i]:
                correct_cnt += 1
        return correct_cnt / test.shape[0]

    def predict(self, data):
        diff = self.model_data - data
        distance = np.linalg.norm(diff, axis=1, keepdims=True)
        sort_index = distance.argsort(axis=0)[:self.K]
        # print("________sorted_index:\n", sort_index)
        selected_label = self.model_label[sort_index][:, 0]
        print("________selected_label:\n", selected_label)
        k_pool = list()
        for label in enumerate(selected_label):
            k_pool.append(str(label))
        result = Counter(k_pool).most_common(1)[0]
        print("result: ", result[0])
        return result[0]


def run_with_KFold(data, K_split, K_near, use_fisher, Log):
    model = KnnModel()
    accuracy = utils.Kfold_model(data, K_split, K_near, use_fisher, model, Log)
    print("average accuracy:", accuracy)
