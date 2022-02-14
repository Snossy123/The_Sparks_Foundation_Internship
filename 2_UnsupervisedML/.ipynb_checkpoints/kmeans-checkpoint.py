import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum( (x1-x2) ** 2) )


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps 
         

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K+1)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.data = X
        self.n_samples, self.m_features = X.shape
        
        # initialize
        centroidsIndex = np.random.choice(self.n_samples, self.K, replace = False)

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            old_centroid = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(old_centroid, self.centroids): 
                break

            if self.plot_steps:
                self.plot()
            
        # Get Inertia In Case determin K Clusters 
        self.Inertia = _get_Inertia(self.clusters, self.centroids)
            
        # Classify samples as the index of their clusters
         return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        
        return labels

         

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.data):
            closest_centroid_idx = _closest_centroid(sample, centroids)
            clusters[closest_centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(point, sample) for point in enumerate(centroids)]
        closest_centroid = np.argmin(distances)
        return closest_centroid
            
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = [np.mean(self.data[c], axis=0) for c in enumerate(clusters)]
        return centroids
        
    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
        
    def plot(self):
        fig, ax = plt.subplots()
        for i, indx in enumerate(self.clusters):
            point = self.data[indx].T
            ax.scatter(*point)
        
        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=2)

        plt.show()
        
    def _get_Inertia(self, clusters, centroids):
        WCSS = 0
        
        for i, indx in enumerate(self.clusters):
            WCSS_Cluster = 0
            for point in self.data[indx]
               WCSS_Cluster += (point - centroids[i]) ** 2 
            WCSS += WCSS_Cluster
            
        return WCSS
