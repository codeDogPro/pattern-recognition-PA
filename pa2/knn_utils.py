import utils
import numpy as np
from collections import Counter


class KnnModel(object):
    def __init__(self):
        self.model_data, self.model_label = None, None
        self.K = 5

    def train(self, train_data, train_label, K=5):
        self.model_data = np.matrix(train_data, dtype=np.float64)
        self.model_label = train_label
        self.K = K

    def train_fisher(self, train_data, train_label, K=5):
        self.K = K

    def validation(self, test_data, test_label):
        test_data = np.matrix(test_data, dtype=np.float64)
        correct_cnt = 0
        for i in range(test_data.shape[0]):
            result = self.predict(test_data[i])
            if result == str(test_label[i]):
                correct_cnt += 1
        return correct_cnt / test_data.shape[0]

    def predict(self, data):
        diff = self.model_data - data
        distance = np.linalg.norm(diff, axis=1, keepdims=True)
        sort_index = distance.argsort(axis=0)[:self.K]
        selected_label = self.model_label[sort_index][:, 0]
        # print("________sorted_index:\n", sort_index)
        # print("________selected_label:\n", selected_label)
        k_pool = list()
        for i in range(selected_label.shape[0]):
            k_pool.append(str(selected_label[i]))
        result = Counter(k_pool).most_common(1)[0][0]
        return result


def run_with_KFold(data, K_split, K_near, use_fisher, Log):
    model = KnnModel()
    accuracy = utils.Kfold_model(data, K_split, K_near, use_fisher, model, Log)
    print("average accuracy:", accuracy)
