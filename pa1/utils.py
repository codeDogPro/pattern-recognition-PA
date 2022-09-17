import pandas as pd


def load_dataset(path):
    data = pd.read_csv(path)
    print(data)
    return data


def fisher(data):
    pass


def validation(data):
    pass


def run_in_Kfold(data, K):
    block_sz = len(data) / K
    for i in range(K):
        test = data[i * block_sz: (i + 1) * block_sz]
        train = data[0: i * block_sz]
        train.extend(data)

        fisher(train)       # train data

        validation(test)    # test the model

