import argparse
import utils
import fisher_utils

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

    data = utils.load_dataset(args.dataset)
    fisher_utils.run_by_Kfold(data, args.K)


if __name__ == '__main__':
    main()
