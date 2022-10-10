import argparse
from knn_utils import *


def main():
    parser = argparse.ArgumentParser(description='knn classifier')
    parser.add_argument('dataset',      type=str,                 help='path of dataset.')
    parser.add_argument('--K_split',    type=int,  default=5,     help='Kfold K\'s value.')
    parser.add_argument('--K_near',     type=int,  default=5,     help='KNN K\'s value.')
    parser.add_argument('--use_fisher', type=bool, default=False, help='use fisher to reduce dimension')
    parser.add_argument('--Log',        type=bool, default=False, help='Log much things')
    args = parser.parse_args()

    print(args.dataset, args.K_split, args.K_near, args.use_fisher, args.Log)
    data = utils.load_dataset(args.dataset)
    if args.K_split:
        run_with_KFold(data, args.K_split, args.K_near, args.use_fisher, args.Log)
    else:
        # for usps dataset that not easy to use KFold
        train_data, train_label = utils.sprt_h5(data[0])
        test_data, test_label = utils.sprt_h5(data[1])
        model = KnnModel()
        model.train(train_data, train_label)
        accuracy = model.validation(test_data, test_label)
        print("accuracy:", accuracy)


if __name__ == '__main__':
    main()
