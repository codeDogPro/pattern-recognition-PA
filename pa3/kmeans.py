import numpy as np


class Cluster:
    def __init__(self, center):
        self.center = center
        self.ndot = 1
        self.dot_list = [center]

    def add(self, dot):
        self.dot_list.append(dot)
        self.ndot += 1

    def remove(self, dot):
        for i in range(self.ndot):
            if (self.dot_list[i] == dot).all():
                self.dot_list.pop(i)
                self.ndot -= 1
                break


def get_newcenter(dots: list[np.ndarray]) -> np.ndarray:
    return np.average(np.array(dots), axis=0)


def kmeans(data: np.ndarray, k: int, epoch: int = 10) -> list[Cluster]:
    kcluster, data_idx = [], np.zeros(data.shape[0], dtype=int) - 1
    ndata = data.shape[0]
    for i in range(k):
        idx = np.random.randint(0, ndata)
        kcluster.append(Cluster(data[idx, :]))

    for j in range(epoch):
        flag = True
        for idx, dot in enumerate(data):
            distance = np.ndarray(k)
            # print(distance)
            for i in range(k):
                if (dot == kcluster[i].center).all():
                    continue
                distance[i] = np.linalg.norm(dot - kcluster[i].center, ord=2)
            nearest_idx = distance.argmin(axis=-1)
            if data_idx[idx] != -1:
                kcluster[data_idx[idx]].remove(dot)  # remove previous class
            kcluster[nearest_idx].add(dot)
            data_idx[idx] = nearest_idx

        for i in range(k):
            new_center = get_newcenter(kcluster[i].dot_list)
            if (new_center == kcluster[i].center).all() is not True:
                flag = False
            kcluster[i].center = new_center

        if flag:  # if all the center aren't changed, stop the algorithm
            break

    for i in range(k):
        # print(f'list:{kcluster[i].ndot}')
        kcluster[i].dot_list = np.array(kcluster[i].dot_list)
        # print(kcluster[i].dot_list.shape)
    return kcluster


def fcm():
    pass
