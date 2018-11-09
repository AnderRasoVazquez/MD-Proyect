import sys
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
    def test():
        data_path = "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/verbal_autopsies_clean.csv"
        output_folder = "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files"
        k = 96
        tolerance = 0.1
        m = 2
        init_strat = 'random'
        max_it = 50

        data = pd.read_csv(data_path, header=0)
        kmeans = KMeans(output_folder, data=data, k=k, tolerance=tolerance, m=m, init_strat=init_strat, max_it=max_it)
        return kmeans

    @staticmethod
    def main():
        try:
            data_path = sys.argv[1]
            output_folder = sys.argv[2]
            k = int(sys.argv[3])
            tolerance = float(sys.argv[4])
        except IndexError:
            # TODO rewrite doc.
            documentation = "Reformatea un archivo .cvs para eliminar los saltos de linea en campos de tipo string" \
                            "en la posición que puedan causar problemas.\n" \
                            "Dos argumentos esperados:\n" \
                            "\t1 - Ruta del archivo .csv que se quiere limpiar.\n" \
                            "\t2 - Ruta del archivo .csv en el que guardar el resultado.\n" \
                            "Ejemplo: python csv_cleaner.py file.csv file_clean.csv"
            print(documentation)
            sys.exit(1)

        try:
            m = int(sys.argv[5])
        except IndexError:
            m = 2

        try:
            init_strat = sys.argv[6]
        except IndexError:
            init_strat = KMeans.INIT_RANDOM

        try:
            max_it = int(sys.argv[7])
        except IndexError:
            max_it = 50

        try:
            verbose = sys.argv[8]
        except IndexError:
            verbose = True

        data = pd.read_csv(data_path, header=0)
        if verbose:
            print("loaded data")
        kmeans = KMeans(output_folder, data=data, k=k, tolerance=tolerance, m=m, init_strat=init_strat, max_it=max_it)
        kmeans.form_clusters(verbose=True)
        if verbose:
            print("Finished clustering. Saving results...")
        kmeans.save_clusters(sorted=True)
        if verbose:
            print("Clusters Saved")
        kmeans.save_centroids()
        if verbose:
            print("Centroids Saved")
        kmeans.save_evaluation()
        if verbose:
            print("Evaluation Saved")
            print("Finished")
        return kmeans

    def __init__(self, output_folder, data, k=10, tolerance=0.1, m=2, w2v_strat='tfidf', init_strat='random', max_it=50):
        """Constructor de la clase KMeans.

        output_folder: directorio donde se guardarán los resultados.
        data: DataFrame con los datos a agrupar.
        k: número de clusters a crear.
        tolerance: distancia mínima que tienen que moverse ál menos un cluster para continuar con el algortimo.
        m: parámetro para la distancia de Minkowski.
        w2v_strat: método para convertir los textos a valores numéricos ('tfidf', 'word_embbedings').
        init_strat: método para inicializar los centroides ('random', 'foo').
        max_it: número máximo de iteraciones a realizar, independientemente de la tolerancia.
        """
        self._data = data
        self._instances = np.empty(len(self._data), dtype='object')
        self._k = min(k, len(self._data))
        self._centroids = np.empty(self._k, dtype='object')
        self._centroid_instances = np.full(k, 0, dtype='int64')
        self._prev_centroids = None
        self._belonging_bits = np.full((len(self._data), self._k), 0, dtype='int8')
        self._tolerance = tolerance
        self._distance = MDistance(m)
        self._max_it = max_it
        self._w2v_strat = w2v_strat
        self._init_strat = init_strat.lower()
        self._output_folder = output_folder
        self._pca = None
        self._ready_to_save = False

    def form_clusters(self, verbose=False):
        """Inicia el proceso de clustering."""

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

        self._ready_to_save = True

    def _load_instances(self, w2v_strat, attribute):
        """Convierte las instancias a datos para utilizan en el clustering."""

        if w2v_strat == self.TFIDF:
            wv_matrix = utils.tfidf_filter(self._data, attribute)
            matrix = wv_matrix.A
            for i in range(len(matrix)):
                self._instances[i] = matrix[i]
        elif w2v_strat == self.WEMBEDDINGS:
            self._instances = utils.word_embeddings(self._data, attribute)

    def _initialize_centroids(self, init_strat):
        """Inicializa los centroides."""

        # default: INIT_RANDOM
        return {
            KMeans.INIT_RANDOM: self._init_random_centroids,
            KMeans.INIT_FOO: self._init_foo
        }.get(init_strat, KMeans.INIT_RANDOM)()

    def _init_random_centroids(self):
        """Inicializa los centroides de forma aleatoria."""

        used = [-1]
        for i in range(self._k):
            index = -1
            while index in used:
                index = random.randrange(0, len(self._instances))
            self._centroids[i] = self._instances[index].copy()
            used.append(index)


    # placeholder for future initialization methods
    def _init_foo(self):
        """Inicializa los centroides de forma foo."""

        pass

    def _check_finished(self):
        """Comprueba si los centroides se han desplazado los suficiete como para continuar."""

        if self._prev_centroids is None:
            return False

        for i in range(self._k):
            centroid = self._centroids[i]
            prev = self._prev_centroids[i]
            if self._distance.distance(centroid, prev) > self._tolerance:
                return False

        return True

    def _update_belonging_bits(self):
        """Actualiza la matriz de bits de pertenencia."""

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
        """Actualiza los centroides."""

        self._prev_centroids = self._centroids.copy()
        self._centroids = self._instances.dot(self._belonging_bits) / np.sum(self._belonging_bits, axis=0)

    def save_clusters(self, sorted=False):
        """Guarda los clusters obtenidos.

        Los centroides se guardan en un archivo csv en el mismo formato
        que las instancias originales con un atributo nuevo que indica
        a que cluster pertenece cada instancia.

        Si el parámetro sorted es True, la instancias se ordenarán según
        su cluster.
        """

        if not self._ready_to_save:
            return False

        output_path_clusters = os.path.join(self._output_folder, 'clusters.csv')

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

    def save_centroids(self):
        """Guarda los centroides obtenidos

        Los centroides se guardan en un archivo binario npy.
        """

        if not self._ready_to_save:
            return False

        output_path_centroids = os.path.join(self._output_folder, 'centroids')
        np.save(output_path_centroids, self._centroids)

    def save_evaluation(self):
        """Guarda la evaluación del modelo entrenado.

        Para la evaluación se utiliza el criterio SSE.
        La evaluación se guarda en dos archivos.

        - Un archivo txt en el que se muestra la cohesión total de los clusteres
        y, después, por cada cluster, cuantas instancias contiene, el SSE total
        del cluster y el SSE medio por cada instancia.

        - Un archivo csv en el que se almacena por cada cluster, cuantas instancias
        contiene, el SSE total del cluster y el SSE medio por cada instancia.
        """

        if not self._ready_to_save:
            return False

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
        """Calcula el SSE de un cluster."""

        sse = 0
        centroid = self._centroids[cluster_index]
        for t in range(len(self._instances)):
            if self._belonging_bits[t][cluster_index]:
                sse += self._distance.distance(self._instances[t], centroid) ** 2
        return sse

    def plot(self, indices=None, separate=False, tags=False):
        """Representa los clusteres en un plano cartesiano.

        Solo se dibujarán los clusteres cuyo indice apaecezca en el
        parámetro indices.

        En el plano aparecerán todas las instancias de los clusteres a
        dibujar. Las instancias que pertenecen al mismo cluster tendrán
        el mismo color.

        Si separate en True, se crea un gráfico por cada cluster en lugar
        de dibujarlos todos en el mismo gráfico.

        Para representar las instancias en dos dimensiones se les
        aplica primero en filtro PCA.

        Si el parámetro tags es True, a cada instancia se le añade una
        etiqueta con el valor de su clase.
        """

        if not self._ready_to_save:
            return False

        if indices is None:
            indices = range(len(self._centroids))

        if self._instances is not None:
            if self._pca is None:
                tmp_instances = []
                for centroid in self._centroids:
                    tmp_instances.append(list(centroid))
                for instance in self._instances:
                    tmp_instances.append(list(instance))
                self._pca = utils.pca_filter(tmp_instances, 2)
            for i in range(len(self._centroids)):
                if i in indices:
                    if separate:
                        plt.figure(i)
                    c = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]]
                    plt.scatter(self._pca[i][0], self._pca[i][1], c=c, marker='P')
                    for t in range(len(self._instances)):
                        if self._belonging_bits[t][i]:
                            plt.scatter(self._pca[t + self._k][0], self._pca[t + self._k][1], c=c, marker='.')
                            if tags:
                                plt.text(self._pca[t + self._k][0], self._pca[t + self._k][1], s=self._data['gs_text34'][t], fontsize=10)
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
        return self._centroids.copy()

    def bits(self):
        return self._belonging_bits.copy()

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
    KMeans.main()
