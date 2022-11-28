from sklearn.utils import shuffle
from itertools import combinations
from utils import *
import numpy as np


def cal_Si(data, u):
    """
    calculate the Si matrix and return it.
    """
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
    Sw = np.matrix(S1 + S2)

    W = Sw.I * np.matrix(u1 - u2).T
    W0 = (u1 * W + u2 * W) / 2
    return [W, W0]


def train_fisher(data):
    """
    This function is used to train data with multiple categories
    :param data: data map with 'n' categories
    :return: discriminant function set   .eg  [[w*_1,w0_1], [w*_2, w0_2],.....]
    """
    train_map = list(combinations(range(len(data)), 2))

    skip_label = data[0].shape[1] - 1
    model = []
    for i in range(len(train_map)):
        data1 = data[train_map[i][0]].iloc[:, 0: skip_label].values
        data2 = data[train_map[i][1]].iloc[:, 0: skip_label].values
        model.append([train_map[i], fisher(data1, data2)])
        # print(f"____models____:\n{model[i][1][0]}")
        # print(len(model[i][1][0]))
    return model


def validation(cate_map, model, data):
    """
    validate the model by test_data, which divede by k-fold.
    it will return the correct rate.
    :param cate_map: the categories map that the data have.
                     eg. [[0] = "cate1", [1] = "cate2", ....].
    :param model:  model that to be tested.
    :param data:  the test_data.
    :return: the correct rate
    """
    correct_num = bad_num = 0
    label = data.loc[:, 'label'].values
    data_r = data.iloc[:, 0: data.shape[1] - 1].values

    vote_map = np.zeros(len(model) + 1)
    # print(cate_map)
    data_num = data_r.shape[0]
    for i in range(data_num):
        for j in range(len(model)):
            res = data_r[i] * model[j][1][0] - model[j][1][1]
            category = model[j][0][0] if res > 0 else model[j][0][1]
            vote_map[category] += 1
        ans = vote_map.argmax()
        # print(f"predict:{cate_map[ans]}  label:{label[i]}")
        if vote_map.max() == len(model) - 2:
            bad_num += 1
        if cate_map[ans] == label[i]:
            correct_num += 1
        # else:
            # print("wrong!")
        vote_map.fill(0)
    # print("bad_num:", bad_num)
    return correct_num / data_num


def run_by_Kfold(data, K):
    """
    this function is used to train model,
    validate the model with k-fold cross validation.
    :param data: dataset to be used.
    :param K: num of k.
    """
    assert(K != 0)

    block_sz = int(data.shape[0] / K)
    data = shuffle(data)
    total_correct_rate = 0
    for i in range(K):
        train = pd.concat([data[0: i * block_sz], data[(i + 1) * block_sz:]])
        classified_train, cate_map = data_classify(train)
        model = train_fisher(classified_train)    # train the model

        test = data[i * block_sz: (i + 1) * block_sz]
        total_correct_rate += validation(cate_map, model, test)
        print(total_correct_rate / (i + 1))
    total_correct_rate /= K
    print("Total correct rate: ", total_correct_rate)
