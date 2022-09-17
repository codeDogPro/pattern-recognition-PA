import numpy as np
import argparse
from utils import *


def main():
    parser = argparse.ArgumentParser(description='Fisher classifier')
    parser.add_argument(
        'dataset',
        type=str,
        help='path of dataset.'
    )
    parser.add_argument(
        'K',
        type=int,
        help='K\'s value.'
    )
    args = parser.parse_args()

    print(args.dataset, args.K)

    data = load_dataset(args.dataset)
    run_in_Kfold(data, args.K)


if __name__ == '__main__':
    main()
