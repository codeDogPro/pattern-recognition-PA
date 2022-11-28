from kmeans import kmeans, fcm
import argparse
from utils import load_dataset, use_pca, show_result_plot


def main():
    parser = argparse.ArgumentParser(description='kmeans classifier')
    parser.add_argument('dataset', type=str,             help='path of dataset.')
    parser.add_argument('k',       type=int,  default=5, help='kmeans k\'s value.')
    args = parser.parse_args()
    # print(args.dataset, args.k)

    dataset = load_dataset(args.dataset).values
    data, label = dataset[:, :4], dataset[:, 4:]

    clusters = kmeans(data, args.k)
    import numpy as np
    all_dots, colors = np.array(None), None
    for i, cluster in enumerate(clusters):
        decomp_result = use_pca(cluster.dot_list, ndim=2)
        # print(f"class{i}:\n{decomp_result}")
        if i != 0:
            np.append(decomp_result)

    print(all_dots)
    # show_result_plot(decomp_result)


if __name__ == '__main__':
    main()
