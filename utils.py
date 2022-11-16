import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def load_dataset(path: str) -> pd.DataFrame:
    """
    :param path: dataset path
    :return: loaded data
    """
    assert path is not None
    if path.find('.csv') != -1:
        data = pd.read_csv(path)
    elif path.find('.h5'):
        data = load_h5(path)
    return data


def load_h5(path):
    import h5py
    hf = h5py.File(path, 'r')
    train = hf.get('train')
    test = hf.get('test')
    data = [train, test]
    # print(train.get('data')[:])
    return data


def sprt_h5(data):
    return data.get('data')[:], data.get('target')[:]


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
    classified_data, cate_map = [], []

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
    return classified_data, cate_map


def Kfold_model(data, K_split=5, K_near=5, use_fisher=False, model_class=None, Log=False):
    """
    This method use K-fold cross validation to validate the model
    :param data: dataset
    :param K_split:  kfold's k values
    :param K_near: knn's k values
    :param use_fisher: wether to use fisher to reduce data's dimension
    :param model_class: model that contains train and validation methods
    :param Log: wether to log everything
    :return: average acurracy
    """
    assert data is not None

    kf = KFold(K_split, shuffle=True)
    total_accuracy = 0
    for train_index, test_index in kf.split(data):
        train, test = data.values[train_index], data.values[test_index]
        label_index = train.shape[1]
        train_data, train_label = train[:, :label_index - 1], train[:, label_index - 1:]
        test_data, test_label = test[:, :label_index - 1], test[:, label_index - 1:]

        if use_fisher:
            model_class.train_fisher(train_data, train_label, K_near)
        else:
            model_class.train(train_data, train_label, K_near)
        accuracy = model_class.validation(test_data, test_label)
        total_accuracy += accuracy
        if Log:
            print("currect rate:", accuracy)
    return total_accuracy / K_split


def show_result_plot(data: np.ndarray):
    plt.plot(data)
    plt.show()
