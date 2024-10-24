import numpy as np

class KMeans:

    def __init__(self, X, n_clusters, max_iter=300, tol=1e-4):
        self.X = X
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    def mean(self, value):
        """
        Calculate mean of the dataset column-wise.
        Do not use built-in functions

        :param value: data
        :return the mean value
        """
        
        n = len(value)
        m = len(value[0])
        
        mean_values = [0] * m
        
        for i in range(n):
            for j in range(m):
                mean_values[j] += value[i][j]
        
        for j in range(m):
            mean_values[j] /= n
        
        return mean_values

    def std(self):
        """
        Calculate standard deviation of the dataset.
        Use the mean function you wrote. Do not use built-in functions

        :param X: dataset
        :return the standard deviation value
        """
        
        mean_values = self.mean(self.X)
        n = len(self.X)
        m = len(self.X[0])
        
        std_values = [0] * m
        
        for i in range(n):
            for j in range(m):
                std_values[j] += (self.X[i][j] - mean_values[j]) ** 2
        
        for j in range(m):
            std_values[j] = (std_values[j] / n) ** 0.5
        
        return std_values

    def standard_scaler(self):
        """
        Implement Standard Scaler to X.
        Use the mean and std functions you wrote. Do not use built-in functions

        :param X: dataset
        :return X_scaled: standard scaled X
        """
        
        mean_values = self.mean(self.X)
        std_values = self.std()
        n = len(self.X)
        m = len(self.X[0])
        
        X_scaled = [[0] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                X_scaled[i][j] = (self.X[i][j] - mean_values[j]) / std_values[j]
    
        return X_scaled

    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two data points.
        Do not use any external libraries

        :param point1: data point 1, list of floats
        :param point2: data point 2, list of floats

        :return the Euclidean distance between two data points
        """
        
        distance = 0
        
        if len(point1) != len(point2):
            raise Exception("Dimension of point must be equal.")
        
        for i in range(len(point1)):
            distance += (point2[i] - point1[i])**2
            
        return distance**(1/2)

    def get_closest_centroid(self, point):
        """
        Find the closest centroid given a data point.

        :param point: list of floats
        :param centroids: a list of list where each row represents the point of each centroid
        :return: the number(index) of the closest centroid
        """
        
        min_distance = float('inf')
        closest_centroid = -1
        
        for i in range(len(self.centroids)):
            distance = self.euclidean_distance(point, self.centroids[i])
            if distance < min_distance:
                min_distance = distance
                closest_centroid = i
                
        return closest_centroid

    def update_clusters(self):
        """
        Assign all data points to the closest centroids.
        Use "get_closest_centroid" function

        :return: cluster_dict: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid
        Example:
        list_of_points = [[1.1, 1], [4, 2], [0, 0]]
        centroids = [[1, 1],
                    [2, 2]]

            print(update_clusters())
        Output:
            {'0': [[1.1, 1], [0, 0]],
             '1': [[4, 2]]}
        """
        
        clusters = {str(i): [] for i in range(self.n_clusters)}
    
        for point in self.X:
            closest_centroid = self.get_closest_centroid(point)
            clusters[str(closest_centroid)].append(point)
        
        return clusters

    def update_centroids(self, cluster_dict):
        """
        Update centroids using the mean of the given points in the cluster.
        Doesn't return anything, only change self.centroids
        Use your mean function.
        Consider the case when one cluster doesn't have any point in it !

        :param cluster_dict: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid
        """
        
        for key in cluster_dict:
            if cluster_dict[key]:  # If the cluster is not empty
                self.centroids[int(key)] = self.mean(cluster_dict[key])

    def converged(self, clusters, old_clusters):
        """
        Check the clusters converged or not

        :param clusters: new clusters, dictionary where keys are cluster labels and values are the points(list of list)
        :param old_clusters: old clusters, dictionary where keys are cluster labels and values are the points(list of list)
        :return: boolean value: True if clusters don't change
        Example:
        clusters = {'0': [[1.1, 1], [0, 0]],
                    '1': [[4, 2]]}
        old_clusters = {'0': [[1.1, 1], [0, 0]],
                        '1': [[4, 2]]}
            print(update_assignment(clusters, old_clusters))
        Output:
            True
        """
        
        for key in clusters:
            if set(tuple(point) for point in clusters[key]) != set(tuple(point) for point in old_clusters[key]):
                return False
        return True

    def calculate_wcss(self, clusters):
        """
        :param clusters: dictionary where keys are clusters labels and values the data points belong to that cluster
        :return:
        """
        
        wcss = 0
        for key in clusters:
            centroid = self.centroids[int(key)]
            for point in clusters[key]:
                wcss += self.euclidean_distance(centroid, point) ** 2
        return wcss

    def fit(self):
        """
        Implement K-Means clustering until the clusters don't change.
        Use the functions you have already implemented.
        Print how many steps does it take to converge.
        :return: final_clusters: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid
                 final_centroids: list of list with shape (n_cluster, X.shape[1])
                 wcss: within-cluster sum of squares
        """
        
        for i in range(self.max_iter):
            clusters = self.update_clusters()
            old_centroids = self.centroids.copy()
            self.update_centroids(clusters)
            if self.converged(clusters, self.update_clusters()):
                print(f'Converged in {i + 1} steps')
                break
        else:
            print(f'Max iterations reached: {self.max_iter}')

        final_clusters = self.update_clusters()
        wcss = self.calculate_wcss(final_clusters)
        return final_clusters, self.centroids, wcss
