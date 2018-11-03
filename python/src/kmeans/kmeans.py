import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from src.utils import utils
from src.utils.distance import MDistance


class KMeans:
    INIT_RANDOM = 'random'
    INIT_FOO = 'foo'  # placeholder for future initialization methods

    TFIDF = 'tfidf'
    WEMBEDDINGS = 'word_embeddings'

    @staticmethod
    def main(data_path, output_folder, k, tolerance=0.1, m=2, init_strat=INIT_RANDOM, max_it=50, verbose=True):
        data = pd.read_csv(data_path, header=0)
        print("loaded data")
        kmeans = KMeans(output_folder, data, k=k, tolerance=tolerance, m=m, init_strat=init_strat, max_it=max_it)
        kmeans.form_clusters(verbose=True)
        if verbose:
            print("Finished clustering. Saving results...")
        kmeans.save_clusters(sorted=True)
        if verbose:
            print("Clusters Saved")
        kmeans.save_evaluation()
        if verbose:
            print("Evaluation Saved")
        # if verbose:
        #     print("Plotting Results...")
        # kmeans.plot(separate=False)
        if verbose:
            print("Finished")
        return kmeans

    def __init__(self, output_folder, data, k=10, tolerance=0.1, m=2, w2v_strat='tfidf', init_strat='random', max_it=50):
        self._data = data
        self._instances = np.empty(len(data), dtype='object')
        self._k = min(k, len(self._data))
        self._centroids = np.empty(k, dtype='object')
        self._centroid_instances = np.full(self._k, 0, dtype='int64')
        self._prev_centroids = None
        self._belonging_bits = np.full((len(self._data), self._k), 0, dtype='int8')
        self._tolerance = tolerance
        self._distance = MDistance(m)
        self._max_it = max_it
        self._w2v_strat = w2v_strat
        self._init_strat = init_strat.lower()
        self._output_folder = output_folder

    def form_clusters(self, verbose=False):
        self._load_instances(w2v_strat=self._w2v_strat, attribute='open_response')
        self._initialize_centroids(self._init_strat)
        it = 0

        while it < self._max_it and not self._check_finished():
            it += 1
            if verbose:
                print("It: {}".format(it))
            self._update_belonging_bits()
            self._update_centroids()

        self._update_belonging_bits()

    def _load_instances(self, w2v_strat, attribute):
        if w2v_strat == self.TFIDF:
            wv_matrix = utils.tfidf_filter(self._data, attribute)
            matrix = wv_matrix.A
            for i in range(len(matrix)):
                self._instances[i] = matrix[i]
        elif w2v_strat == self.WEMBEDDINGS:
            self._instances = utils.word_embeddings(self._data, attribute)

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
        self._belonging_bits = np.full((len(self._data), self._k), 0, dtype='int8')
        self._centroid_instances = np.full(self._k, 0, dtype='int64')

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

            self._belonging_bits[t][min_i] = 1
            self._centroid_instances[min_i] += 1

    def _update_centroids(self):
        self._prev_centroids = self._centroids.copy()
        self._centroids = self._instances.dot(self._belonging_bits) / np.sum(self._belonging_bits, axis=0)

    def save_clusters(self, sorted=False):
        output_path_clusters = os.path.join(self._output_folder, 'clusters.csv')
        output_path_centroids = os.path.join(self._output_folder, 'centroids')

        results = self._data.copy()
        cluster_col = []
        for t in range(len(self._instances)):
            for i in range(self._k):
                if self._belonging_bits[t][i]:
                    cluster_col.append(i)
                    break

        results['cluster'] = cluster_col
        cols = list(results)
        cols.insert(0, cols.pop(cols.index('cluster')))
        results = results.ix[:, cols]
        if sorted:
            results = results.sort_values(['cluster', 'gs_text34'])
        results.to_csv(output_path_clusters, index=False)
        np.save(output_path_centroids, self._centroids)

    def save_evaluation(self):
        output_path1 = os.path.join(self._output_folder, 'evaluation.txt')
        output_path2 = os.path.join(self._output_folder, 'evaluation.csv')
        cohesion = 0
        clusters_sse = ''

        dataframe_columns = ['cluster', 'n_instances', 'sse', 'sse_avg']
        dataframe = pd.DataFrame(columns=dataframe_columns)
        for i in range(len(self._centroids)):
            sse_i = self._sse(i)
            n_instances = self._centroid_instances[i]
            cohesion += sse_i
            clusters_sse += 'CLUSTER {} ({} instances) -> SSE = {} // {} avg.\n'.\
                format(i, n_instances, sse_i, sse_i/n_instances)

            dataframe = dataframe.append(pd.DataFrame([[i, n_instances, sse_i, sse_i/n_instances]],
                                          columns=dataframe_columns),
                             ignore_index=True)
        with open(output_path1, 'w') as f:
            f.write("TOTAL COHESION = {}\n".format(cohesion))
            f.write(clusters_sse)

        dataframe.to_csv(output_path2, index=False)

    def _sse(self, cluster_index):
        sse = 0
        centroid = self._centroids[cluster_index]
        for t in range(len(self._instances)):
            if self._belonging_bits[t][cluster_index]:
                sse += self._distance.distance(self._instances[t], centroid) ** 2
        return sse

    def plot(self, separate=False):
        if self._instances is not None:
            tmp_instances = []
            for instance in self._instances:
                tmp_instances.append(list(instance))
            pca = utils.pca_filter(tmp_instances, 2)
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

    def data(self):
        return self._data.copy()

    def instances(self):
        return self._instances.copy()

    def k(self):
        return self._k

    def set_k(self, k):
        self._k = k

    def centroids(self):
        return self._centroids

    def bits(self):
        return self._belonging_bits

    def tolerance(self):
        return self._tolerance

    def set_tolerance(self, tolerance):
        self._tolerance = tolerance

    def distance_m(self):
        return self._distance.m()

    def set_distance_m(self, m):
        return self._distance.set_m(m)

    def max_it(self):
        return self._max_it

    def set_max_it(self, max_it):
        self._max_it = max_it

    def w2v_strat(self):
        return self._w2v_strat

    def set_w2v_strat(self, w2v_strat):
        self._w2v_strat = w2v_strat

    def init_strat(self):
        return self._init_strat

    def set_init_strat(self, init_strat):
        self._init_strat = init_strat

    def output_folder(self):
        return self._output_folder

    def set_output_folder(self, output_folder):
        self._output_folder = output_folder


if __name__ == '__main__':
    KMeans.main("/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/verbal_autopsies_clean.csv",
                "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files",
                k=96, m=2)
