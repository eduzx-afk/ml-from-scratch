import numpy as np


class KMeansDIY:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, x, max_iter=50):
        self.cluster_centers_ = np.random.rand(self.n_clusters, x.shape[1])

        for iter in range(max_iter):
            old_centres = self.cluster_centers_.copy()
            self.change_centre(x)

            if np.allclose(old_centres, self.cluster_centers_):
                break

    def predict(self, x):
        return np.array([
            self.find_closest_centre(sample)
            for sample in x
        ])

    def calc_distance(self, x, c):
        return np.sqrt(np.sum((x - c) ** 2))

    def find_closest_centre(self, x):
        closest_distance = float("inf")
        closest_idx = None

        for cluster in range(self.n_clusters):
            centre = self.cluster_centers_[cluster]
            distance = self.calc_distance(x, centre)

            if distance < closest_distance:
                closest_distance = distance
                closest_idx = cluster

        return closest_idx

    def change_centre(self, x):
        classes = self.predict(x)

        for index in range(self.n_clusters):
            data = x[classes == index]

            if len(data) > 0:
                self.cluster_centers_[index] = np.mean(data, axis=0)
