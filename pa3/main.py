from kmeans import kmeans, fcm
import argparse
from utils import load_dataset, show_result_plot


def main():
    parser = argparse.ArgumentParser(description='kmeans classifier')
    parser.add_argument('dataset', type=str,             help='path of dataset.')
    parser.add_argument('k',       type=int,  default=5, help='kmeans k\'s value.')
    args = parser.parse_args()
    # print(args.dataset, args.k)

    dataset = load_dataset(args.dataset).values
    data, label = dataset[:, :4], dataset[:, 4:]

    clusters = kmeans(data, args.k)
    for i in range(args.k):
        print(clusters[i].dot_list)

    _kmeans = KMeans(n_clusters=3, random_state=123).fit(iris_dataScale)  # 构建并训练模型
    fowlkes_mallows_score()


if __name__ == '__main__':
    main()
