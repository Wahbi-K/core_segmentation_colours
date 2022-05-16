import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class Clusterer:

    def __init__(self, num_clusters=5, clustering_method='KNN'):
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method
        assert self.clustering_method in ('KNN', 'GMM')
        self.instantiate_model()

    def instantiate_model(self) -> None:
        if self.clustering_method == 'KNN':
            self.model = KMeans(n_clusters=self.num_clusters)
        elif self.clustering_method == 'GMM':
            self.model = GaussianMixture(n_components=self.num_clusters)
        else:
            raise ValueError("Choose from GMM or KNN")

    def fit(self, data: np.array) -> None:
        self.model.fit(data)

    def predict(self, data: np.array) -> None:
        return self.model.predict(data)

    @property
    def centroids(self) -> np.array:
        return self.model.cluster_centers_

    def collapse_centroids(self, threshold):
        raise NotImplementedError
