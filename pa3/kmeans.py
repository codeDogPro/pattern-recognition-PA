import numpy as np
import matplotlib as plt


class Cluster:
    def __init__(self, center):
        self.center = center
        self.ndot = 1
        self.dot_list = [center]

    def add(self, dot):
        self.dot_list.append(dot)
        self.ndot += 1


def get_newcenter(dots: list[np.ndarray]) -> np.ndarray:
    return np.average(np.array(dots), axis=0)


def kmeans(data: np.ndarray, k: int, epoch: int = 10) -> list[Cluster]:
    kcluster = []
    ndata = data.shape[0]
    for i in range(k):
        idx = np.random.randint(0, ndata)
        kcluster.append(Cluster(data[idx, :]))

    for j in range(epoch):
        flag = True
        for dot in data:
            dist = np.ndarray(k)
            # print(dist)
            for i in range(k):
                if (dot == kcluster[i].center).all():
                    continue

                dist[i] = np.linalg.norm(dot - kcluster[i].center, ord=2)
            nearest_idx = dist.argmin(axis=-1)
            kcluster[nearest_idx].add(dot)

        for i in range(k):
            new_center = get_newcenter(kcluster[i].dot_list)
            if (new_center == kcluster[i].center).all() is not True:
                flag = False
            kcluster[i].center = new_center

        if flag:  # if all the center aren't changed, stop the algorithm
            break
    return kcluster


def fcm():
    pass
