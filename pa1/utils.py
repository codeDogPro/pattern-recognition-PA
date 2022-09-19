import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from itertools import combinations


def load_dataset(path):
    data = pd.read_csv(path)
    print(data)
    return data


def show_classified_data(data):
    print("______classified_data_______")
    print("category num: " + str(len(data)))
    for i in range(len(data)):
        label_col = data[i].shape[1] - 1
        print("label: " + data[i].iloc[0, label_col])
        print(data[i])


def data_classify(data):
    """"
    data: unclassified data(dataframe)
    return: classified data array, which contains N dataframe
    """
    classified_data = []

    data = data.sort_values(by='label')

    label_col = data.shape[1] - 1
    label = data.iloc[0, label_col]
    l_bound = 0
    for i in range(data.shape[0]):
        if data.iloc[i, label_col] == label:
            continue
        else:
            classified_data.append(data[l_bound: i])
            l_bound = i
            label = data.iloc[i, label_col]
    classified_data.append(data[l_bound:])
    return classified_data


def cal_Si(data, u):
    data -= u
    ndim = data.shape[1]
    matrix = np.zeros((ndim, ndim))
    for i in range(data.shape[0]):
        matrix += np.matrix(data[i]).T * data[i]
    return matrix


def fisher(data1, data2):
    """
    we can't sum before gen_mat, it will make singular matrix
    """
    u1 = data1.mean(axis=0)
    u2 = data2.mean(axis=0)
    S1 = cal_Si(data1, u1)
    S2 = cal_Si(data2, u2)
    # print(f"_____S1_____\n{S1}")
    # print(f"_____S2_____\n{S2}")
    Sw = np.matrix(S1 + S2)
    # print(f"_____Sw_____\n{Sw}")
    W = np.dot(Sw.I, np.matrix(u1 - u2).T)
    # print(f"_____W_____\n{W}")
    W0 = (u1 * W - u2 * W) / 2
    # print(f"_____W0_____\n{W0}")
    return [W, W0]


def train_fisher(data):
    """
    This function is used to train data with multiple categories
    :param data: data map with 'n' categories
    :return: discriminant function
    """
    train_map = list(combinations(range(len(data)), 2))
    # print(train_map)
    skip_label = data[0].shape[1] - 1
    models = []
    for i in range(len(train_map)):
        data1 = data[train_map[i][0]].iloc[:, 0: skip_label].values
        data2 = data[train_map[i][1]].iloc[:, 0: skip_label].values
        models.append(fisher(data1, data2))
    print(len(models))
    return models


def validation(model, data, label):
    pass


def run_in_Kfold(data, K):
    block_sz = int(data.shape[0] / K)
    print(block_sz)
    data = shuffle(data)
    for i in range(K):
        train = pd.concat([data[0: i * block_sz], data[(i + 1) * block_sz:]])
        classified_train = data_classify(train)
        # print("_______train_________")
        # show_classified_data(classified_train)
        model = train_fisher(classified_train)    # train the model

        test = data[i * block_sz: (i + 1) * block_sz]
        classified_test = data_classify(test)
        # print("_______test_________")
        # show_classified_data(classified_test)
        # label1 = data[train_map[i][0]].loc[:, 'label'].values

