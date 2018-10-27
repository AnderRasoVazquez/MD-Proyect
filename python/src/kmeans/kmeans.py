import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from src.utils import utils
from src.utils.distance import MDistance


class KMeans:
    INIT_RANDOM = "random"
    INIT_FOO = "foo"  # placeholder for future initialization methods

    @staticmethod
    def main(data_path, output_path, k, m):
        data = pd.read_csv(data_path, header=0)
        print("loaded data")
        kmeans = KMeans(output_path, data, k=k, m=m)
        kmeans.form_clusters(verbose=True)
        # kmeans.plot(separate=True)

    def __init__(self, output_path, data, k=10, tolerance=0.1, m=2, init_strat="random", max_it=50):
        self._data = data
        self._instances = None
        self._k = min(k, len(self._data))
        self._centroids = [None] * k
        self._centroid_instances = [0] * k
        self._prev_centroids = None
        self._belonging_bits = [[False for i in range(k)] for t in range(len(self._data))]
        self._tolerance = tolerance
        self._distance = MDistance(m)
        self._max_it = max_it
        self._init_strat = init_strat.lower()
        self._output_path = output_path

    def form_clusters(self, verbose=False):
        wv_matrix = utils.tfidf_filter(self._data, 'open_response')
        self._instances = wv_matrix.A

        self._initialize_centroids(self._init_strat)
        it = 0

        while it < self._max_it and not self._check_finished():
            it += 1
            if verbose:
                print("It: {}".format(it))
            self._update_belonging_bits()
            self._update_centroids()

        self._update_belonging_bits()
        if verbose:
            print("Finished clustering. Saving results...")
        self._save_instances()
        if verbose:
            print("Instances Saved")
        self._save_clusters(verbose)
        if verbose:
            print("Clusters Saved")

    def _initialize_centroids(self, init_strat):
        # default: INIT_RANDOM
        return {
            KMeans.INIT_RANDOM: self._init_random_centroids,
            KMeans.INIT_FOO: self._init_foo
        }.get(init_strat, KMeans.INIT_RANDOM)()

    def _init_random_centroids(self):
        used = [-1]
        for i in range(self._k):
            index = -1
            while index in used:
                index = random.randrange(0, len(self._instances))
            self._centroids[i] = self._instances[index].copy()
            used.append(index)

    # placeholder for future initialization methods
    def _init_foo(self):
        pass

    def _check_finished(self):
        if self._prev_centroids is None:
            return False

        for i in range(self._k):
            centroid = self._centroids[i]
            prev = self._prev_centroids[i]
            if self._distance.distance(centroid, prev) > self._tolerance:
                return False

        return True

    def _update_belonging_bits(self):
        self._belonging_bits = [[False for i in range(self._k)] for t in range(len(self._instances))]

        for t in range(len(self._instances)):
            min_diff = 99999
            min_i = -1
            instance = self._instances[t]

            for i in range(self._k):
                centroid = self._centroids[i]
                diff = self._distance.distance(instance, centroid)

                if (diff < min_diff or
                        (diff == min_diff and
                         self._centroid_instances[i] < self._centroid_instances[min_i])):
                    min_diff = diff
                    min_i = i

            self._belonging_bits[t][min_i] = True
            self._centroid_instances[min_i] += 1

    def _update_centroids(self):
        self._prev_centroids = self._centroids.copy()

        for i in range(self._k):
            bits_i = 0
            instance_sum = None

            for t in range(len(self._instances)):
                if self._belonging_bits[t][i]:
                    bits_i += 1
                    if instance_sum is None:
                        instance_sum = np.array([x if utils.is_number(x) else np.nan for x in self._instances[t].copy()])
                    else:
                        instance_sum = instance_sum + self._instances[t]

            self._centroids[i] = np.divide(instance_sum, bits_i)
            self._centroid_instances[i] = 0

    def _save_instances(self):
        with open(self._output_path, 'w') as f:
            for t in range(len(self._instances)):
                for i in range(self._k):
                    if self._belonging_bits[t][i]:
                        f.write('INSTANCE {} -> CLUSTER {} // {}\n'
                                .format(t, i, str(self._data.get_values()[t]).replace('\n', '')))

    def _save_clusters(self, verbose=False):
        tmp_path = self._output_path
        if tmp_path.endswith('/'):
            tmp_path = tmp_path[:-1]
        tmp_path = '/'.join(tmp_path.split('/')[:-1])
        dir_path = os.path.join(tmp_path, 'clusters')
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass

        for i in range(self._k):
            cluster = pd.SparseDataFrame(columns=self._data.columns)
            for t in range(len(self._instances)):
                if self._belonging_bits[t][i]:
                    instance_data = pd.SparseDataFrame([self._data.get_values()[t]], columns=self._data.columns)
                    cluster = cluster.append(instance_data.copy(), ignore_index=True)

            cluster_path = os.path.join(dir_path, 'cluster{}.csv'.format(i))
            cluster = cluster.sort_values(by=['gs_text34'])
            cluster.to_csv(path_or_buf=cluster_path, index_label=False)
            if verbose:
                print("Saved cluster nº {}".format(i))

    def plot(self, separate):
        if self._instances is not None:
            pca = utils.pca_filter(self._instances, 2)
            for i in range(len(self._centroids)):
                if separate:
                    plt.figure(i)
                c = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]]
                for t in range(len(self._instances)):
                    if self._belonging_bits[t][i]:
                        plt.scatter(pca[t][0], pca[t][1], c=c)
                        if separate:
                            plt.text(pca[t][0], pca[t][1], s=self._data['gs_text34'][t], fontsize=10)
                if separate:
                    plt.title("Cluster {}".format(i))
            plt.show()


if __name__ == '__main__':
    KMeans.main("/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/verbal_autopsies_clean.csv",
                "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/assigned_instances.txt",
                48, 2)
    # KMeans.main("/home/ander/github/MD-Proyect/files/verbal_autopsies_tfidf_s.csv",
    #             "/home/ander/github/MD-Proyect/files/assigned_instances.txt",
    #             48, 2)
