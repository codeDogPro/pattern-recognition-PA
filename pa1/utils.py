import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from itertools import combinations


def load_dataset(path):
    data = pd.read_csv(path)
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
    cate_map = []

    data = data.sort_values(by='label')

    label_col = data.shape[1] - 1
    label = data.iloc[0, label_col]
    cate_map.append(label)
    l_bound = 0
    for i in range(data.shape[0]):
        if data.iloc[i, label_col] == label:
            continue
        else:
            classified_data.append(data[l_bound: i])
            l_bound = i
            label = data.iloc[i, label_col]
            cate_map.append(label)
    classified_data.append(data[l_bound:])
    return [classified_data, cate_map]


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

    skip_label = data[0].shape[1] - 1
    model = []
    for i in range(len(train_map)):
        data1 = data[train_map[i][0]].iloc[:, 0: skip_label].values
        data2 = data[train_map[i][1]].iloc[:, 0: skip_label].values
        model.append([train_map[i], fisher(data1, data2)])
        # print(f"____models____:\n{model[i][1][0]}")
    return model


def validation(cate_map, model, data):
    correct_num = 0
    bad_num = 0

    label = data.loc[:, 'label'].values
    data_r = data.iloc[:, 0: data.shape[1] - 1].values

    vote_map = np.zeros(len(model))
    # print(cate_map)
    data_num = data_r.shape[0]
    for i in range(data_num):
        for j in range(len(model)):
            res = data_r[i] * model[j][1][0] - model[j][1][1]
            category = model[j][0][0] if res > 0 else model[j][0][1]
            # print(f"res:{res} category_predict:{category} label:{label[i]}")
            vote_map[category] += 1
        ans = vote_map.argmax()
        if vote_map.max() == 1: #debug
            bad_num += 1
        if cate_map[ans] == label[i]:
            correct_num += 1
        vote_map.fill(0)
    print("bad_num:", bad_num)
    return correct_num / data_num


def run_by_Kfold(data, K):
    assert(K != 0)

    block_sz = int(data.shape[0] / K)
    data = shuffle(data)
    total_correct_rate = 0
    for i in range(K):
        train = pd.concat([data[0: i * block_sz], data[(i + 1) * block_sz:]])
        data_pkg = data_classify(train)
        classified_train = data_pkg[0]
        model = train_fisher(classified_train)    # train the model

        test = data[i * block_sz: (i + 1) * block_sz]
        cate_map = data_pkg[1]
        total_correct_rate += validation(cate_map, model, test)
        print(total_correct_rate / (i + 1))
    total_correct_rate /= K
    print("Total correct rate: ", total_correct_rate)
