from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture

class Clusterer:

    def __init__(self, num_clusters=5, clustering_method='KNN'):
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method
        assert self.clustering_method in ('KNN', 'GMM')
        self.instantiate_model()

    def instantiate_model(self):
        if self.clustering_method == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=self.num_clusters)
        elif self.clustering_method == 'GMM':
            self.model = GaussianMixture(n_components=self.num_clusters)

    def fit(data):
        self.model.fit(data)

    def predict(data):
        return self.model.predict(data)

    @property
    def centroids(self):
        return self.model.means_

    def collapse_centroids(self, threshold):
        raise NotImplementedError
