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
    run_with_KFold(data, args.K_split, args.K_near, args.use_fisher, args.Log)


if __name__ == '__main__':
    main()
