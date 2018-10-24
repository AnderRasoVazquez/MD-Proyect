import os
import pandas
import numpy as np
import random
from src.utils import utils
from src.utils.distance import MDistance


class KMeans:
    INIT_RANDOM = "random"
    INIT_FOO = "foo"  # placeholder for future initialization methods

    @staticmethod
    def main(data_path, output_path, k, m):
        data = utils.sparse_csv_to_dataframe(data_path, empty_value=0)
        print("loaded data")
        kmeans = KMeans(output_path, data, k=k, m=m)
        kmeans._form_clusters(verbose=True)

    def __init__(self, output_path, data, k=10, tolerance=0.3, m=2, max_it=1, init_strat="random"):
        self._data = data  # no está en forma values
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

    def _form_clusters(self, verbose=False):
        self._instances = self._data.copy().select_dtypes(exclude='object').get_values()
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
                                .format(t, i, utils.vector_to_sparse_string(self._data.get_values()[t], value_to_omit=0)))

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
            cluster = pandas.SparseDataFrame(columns=self._data.columns)

            for t in range(len(self._instances)):
                if self._belonging_bits[t][i]:
                    instance_data = pandas.SparseDataFrame([self._data.get_values()[t]], columns=self._data.columns)
                    cluster = cluster.append(instance_data.copy(), ignore_index=True)

            cluster_path = os.path.join(dir_path, 'cluster{}.csv'.format(i))
            utils.dataframe_to_sparse_csv(cluster, cluster_path, 0)
            if verbose:
                print("Saved cluster nº {}".format(i))


if __name__ == '__main__':
    # KMeans.main("/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/test_s.csv",
    #             "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/assigned_instances.txt",
    #             3, 2)
    KMeans.main("/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/verbal_autopsies_tfidf_s.csv",
                "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/assigned_instances.txt",
                48, 2)
