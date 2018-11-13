import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser
from src.utils import utils
from src.utils.distance import MDistance


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        type=str,
                        help='ruta del archivo csv con las instancias',
                        required=True)
    parser.add_argument('-o', '--output_folder',
                        type=str,
                        help='ruta del directorio en el que guardar los resultados',
                        required=True)
    parser.add_argument('-a', '--text_attribute',
                        type=str,
                        help='atributo texto de las instancias sobre el que'
                             'hacer el clustering',
                        default='open_response')
    parser.add_argument('-y', '--class_attribute',
                        type=str,
                        help='atributo clase de las instancias',
                        default='gs_text34')
    parser.add_argument('-k',
                        type=int,
                        help='número de cluster a crear',
                        default=10)
    parser.add_argument('-m',
                        type=int,
                        help='paŕametro para la distancia Minkowski',
                        default=2)
    parser.add_argument('-t', '--tolerance',
                        type=float,
                        help='distancia mínima que debe variar al menos un centroide'
                             'tras una iteración para continual el algoritmo',
                        default=0.1)
    parser.add_argument('-c', '--inter_cluster_distance',
                        type=str,
                        help='método para calcular la distancia inter-cluster',
                        default='average-link',
                        choices=['average-link', 'single-link', 'complete-link'])
    parser.add_argument('-i', '--init_strategy',
                        type=str,
                        help='método para inicializar los centroides',
                        default='random',
                        choices=['random', '2k'])
    parser.add_argument('-x', '--max_it',
                        type=int,
                        help='número máximo de iteraciones a realizar',
                        default=50)
    parser.add_argument('-p', '--plot_indices',
                        type=list,
                        help='matriz de índices de los clusters a representar'
                             'gráficamente. Por defecto, todos en un solo gráfico',
                        default=None)
    parser.add_argument('-g', '--plot_tags',
                        type=bool,
                        help='flag para incluir etiquetas con las clases de las'
                             'instancias en el gráfico',
                        default=False)
    parser.add_argument('-s', '--plot_save_folder',
                        type=str,
                        help='ruta del directorio en el que guardar los gráficos',
                        default=None)
    parser.add_argument('-v', '--verbose',
                        type=bool,
                        help='flag verbose',
                        default=False)

    return parser.parse_args()


class KMeans:
    INIT_RANDOM = 'random'
    INIT_2K = '2k'

    TFIDF = 'tfidf'
    WEMBEDDINGS = 'word_embeddings'

    AVERAGE_LINK = 'average_link'
    SINGLE_LINK = 'single_link'
    COMPLETE_LINK = 'complete_link'

    @staticmethod
    def test():
        # data_path = "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/verbal_autopsies_clean.csv"
        # output_folder = "/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files"
        # k = 96
        # tolerance = 0.1
        # m = 2
        # inter_cluster_dist = 'single_link'
        # init_strat = 'random'
        # max_it = 50
        #
        # data = pd.read_csv(data_path, header=0)
        # kmeans = KMeans(output_folder, data=data, k=k, tolerance=tolerance,
        #                 m=m, inter_cluster_dist=inter_cluster_dist,
        #                 init_strat=init_strat, max_it=max_it)
        # return kmeans
        pass

    @staticmethod
    def main(data_path, output_folder, text_attr, class_attr, k=10,
             tolerance=0.1, m=2, inter_cluster_dist='average_link',
             init_strat='random', max_it=50, plot_indices=None,
             plot_tags=False, plot_save_folder=None, verbose=False):
        if verbose:
            print("Loading data...")
        data = pd.read_csv(data_path, header=0)
        kmeans = KMeans(output_folder=output_folder, data=data,
                        text_attr=text_attr, class_attr=class_attr,
                        k=k, tolerance=tolerance, m=m,
                        inter_cluster_dist=inter_cluster_dist,
                        init_strat=init_strat, max_it=max_it)
        kmeans.form_clusters(verbose)
        kmeans.save_results(verbose)
        # kmeans.plot(indices_matrix=plot_indices, tags=plot_tags, save_path_folder=plot_save_folder)
        return kmeans

    def __init__(self, output_folder, data, text_attr, class_attr=None,
                 k=10, tolerance=0.1, m=2, inter_cluster_dist='average_link',
                 w2v_strat='tfidf', init_strat='random', max_it=50):
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
        self._text_attr = text_attr
        self._class_attr = class_attr
        self._k = min(k, len(self._data))
        self._centroids = np.empty(self._k, dtype='object')
        self._centroid_instances = np.full(k, 0, dtype='int64')
        self._prev_centroids = None
        self._belonging_bits = np.full((len(self._data), self._k), 0, dtype='int8')
        self._tolerance = tolerance
        self._distance = MDistance(m)
        self._inter_cluster_dist = inter_cluster_dist
        self._max_it = max_it
        self._w2v_strat = w2v_strat
        self._init_strat = init_strat.lower()
        self._output_folder = output_folder
        self._pca = None
        self._ready_to_save = False

    def form_clusters(self, verbose=False):
        """Inicia el proceso de clustering."""

        self._pca = None

        self._load_instances(w2v_strat=self._w2v_strat, attribute=self._text_attr)

        if verbose:
            print("Initializing centroids")

        self._initialize_centroids(self._init_strat)
        it = 0

        if verbose:
            print("Starting clustering")

        while it < self._max_it and not self._check_finished():
            it += 1
            if verbose:
                print("It: {}".format(it))
            self._update_belonging_bits()
            self._update_centroids()

        self._update_belonging_bits()

        self._ready_to_save = True

        if verbose:
            print("Clustering finished")

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
            KMeans.INIT_2K: self._init_2k
        }.get(init_strat, KMeans.INIT_RANDOM)()

    def _init_random_centroids(self):
        """Inicializa los centroides de forma aleatoria."""

        self._pca = None

        used = [-1]
        for i in range(self._k):
            index = -1
            while index in used:
                index = random.randrange(0, len(self._instances))
            self._centroids[i] = self._instances[index].copy()
            used.append(index)

    def _init_2k(self):
        """Inicializa los centroides de a partir de 2k centroides"""

        self._pca = None

        tmp_kmeans = KMeans(output_folder="", data=self.data(), text_attr=self._text_attr,
                            k=2 * self.k(), tolerance=self.tolerance(), m=self.distance_m(),
                            inter_cluster_dist=self.inter_cluster_dist(),
                            init_strat=KMeans.INIT_RANDOM, max_it=5)
        tmp_kmeans._instances = self.instances()
        # se crean los 2k clusters en el nuevo modelo
        tmp_kmeans.form_clusters()
        # se calcula la distancia entre cada par de clusters
        distances = {}
        for i in range(2 * self.k()):
            for j in range(2 * self.k()):
                key = (i, j)
                key2 = (j, i)
                if key not in distances and key2 not in distances:
                    distances[key] = tmp_kmeans._inter_cluster_distance(self._inter_cluster_dist,
                                                                        i, j)
        # de entre los 2k clusters, se buscan los k clusters más separados
        started = False
        used = []
        for i in range(self._k):
            if not started:
                # para el primer cluster se busca aquel cuya suma de todas
                # las distancias al resto de clusters sea mayor
                to_check = range(2 * self._k)
                started = True
            else:
                # para el resto de clusters, se busca aquel cuya suma
                # de las distancias al resto de clusters ya seleccionados
                # sea mayor
                to_check = used
            max_dist = -1
            index = -1
            # se itera sobre todos los clusters no seleccionados
            for j in range(2 * self._k):
                if j not in used:
                    cum_dist = 0
                    # se itera sobre todos los clusters a comparar
                    for j2 in to_check:
                        key = (j, j2)
                        key2 = (j2, j)
                        cum_dist += distances.get(key, distances.get(key2))
                    if cum_dist > max_dist:
                        max_dist = cum_dist
                        index = j
            self._centroids[i] = tmp_kmeans._centroids[index]
            used.append(index)

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

    def save_results(self, verbose=False):
        """Guarda los resultados obtenidos

        Esto incluye los clusters, los centroides y la evaluación.
        """

        if not self._ready_to_save:
            return False

        clusters_path = os.path.join(self._output_folder, 'clusters.csv')
        centroids_path = os.path.join(self._output_folder, 'centroids')
        evaluation_path = os.path.join(self._output_folder, 'evaluation.csv')
        self.save_clusters(clusters_path)
        if verbose:
            print("Save clusters in {}".format(clusters_path))
        self.save_centroids(centroids_path)
        if verbose:
            print("Save centroids in {}".format(centroids_path))
        self.save_evaluation(evaluation_path)
        if verbose:
            print("Save evaluation in {}".format(evaluation_path))

    def save_clusters(self, path, sorted=True):
        """Guarda los clusters obtenidos.

        Los centroides se guardan en un archivo csv en el mismo formato
        que las instancias originales con un atributo nuevo que indica
        a que cluster pertenece cada instancia.

        Si el parámetro sorted es True, la instancias se ordenarán según
        su cluster.
        """

        if not self._ready_to_save:
            return False

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
        if sorted and self._class_attr is not None:
            results = results.sort_values(['cluster', self._class_attr])
        results.to_csv(path, index=False)

    def save_centroids(self, path):
        """Guarda los centroides obtenidos

        Los centroides se guardan en un archivo binario npy.
        """

        if not self._ready_to_save:
            return False

        np.save(path, self._centroids)

    def save_evaluation(self, path):
        """Guarda la evaluación del modelo entrenado.

        Para la evaluación se utiliza el criterio SSE.
        La evaluación se guarda en un archivo csv en el que se almacena,
        por cada cluster, cuantas instancias contiene, el SSE total del
        cluster y el SSE medio por cada instancia.
        """

        if not self._ready_to_save:
            return False

        cohesion = 0
        dataframe_columns = ['cluster', 'n_instances', 'sse', 'sse_avg']
        dataframe = pd.DataFrame(columns=dataframe_columns)
        for i in range(len(self._centroids)):
            sse_i = self._sse(i)
            n_instances = self._centroid_instances[i]
            cohesion += sse_i
            dataframe = dataframe.append(pd.DataFrame([[i, n_instances, sse_i, sse_i/n_instances]],
                                                      columns=dataframe_columns),
                                         ignore_index=True)
        dataframe.to_csv(path, index=False)

    def _sse(self, cluster_index):
        """Calcula el SSE de un cluster."""

        sse = 0
        centroid = self._centroids[cluster_index]
        for t in range(len(self._instances)):
            if self._belonging_bits[t][cluster_index]:
                sse += self._distance.distance(self._instances[t], centroid) ** 2
        return sse

    def _inter_cluster_distance(self, inter_cluster_dist, first_c, second_c):
        """Calcula la distancia inter-cluster"""

        # default: INIT_RANDOM
        return {
            KMeans.AVERAGE_LINK: self._avg_link,
            KMeans.SINGLE_LINK: self._single_link,
            KMeans.COMPLETE_LINK: self._complete_link
        }.get(inter_cluster_dist, KMeans.AVERAGE_LINK)(first_c, second_c)

    def _single_link(self, first_c, second_c):
        """Calcula la distancia single-link entre los clusters"""

        min_dist = 99999
        first_instances = []
        second_instances = []
        for t in range(len(self._instances)):
            if self._belonging_bits[t][first_c]:
                first_instances.append(self._instances[t])
            elif self._belonging_bits[t][second_c]:
                second_instances.append(self._instances[t])

        for a in first_instances:
            for b in second_instances:
                dist = self._distance.distance(a, b)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def _complete_link(self, first_c, second_c):
        """Calcula la distancia complete-link entre los clusters"""

        min_dist = -1
        first_instances = []
        second_instances = []
        for t in range(len(self._instances)):
            if self._belonging_bits[t][first_c]:
                first_instances.append(self._instances[t])
            elif self._belonging_bits[t][second_c]:
                second_instances.append(self._instances[t])

        for a in first_instances:
            for b in second_instances:
                dist = self._distance.distance(a, b)
                if dist > min_dist:
                    min_dist = dist

        return min_dist

    def _avg_link(self, first_c, second_c):
        """Calcula la distancia average-link entre los clusters"""

        return self._distance.distance(self._centroids[first_c], self._centroids[second_c])

    def plot(self, indices_matrix=None, tags=False, save_path_folder=None):
        """Representa los clusteres en un plano cartesiano.

        indices_matrix debe ser un array bidimensional con los índices
        de los centroides que se quieren dibujar. Cada línea de clusters
        se dibujarán en el mismo gráfico, separado del resto de líneas.

        En el plano aparecerán todas las instancias de los clusteres a
        dibujar. Las instancias que pertenecen al mismo cluster tendrán
        el mismo color.

        Para representar las instancias en dos dimensiones se les
        aplica primero en filtro PCA.

        Si el parámetro tags es True, a cada instancia se le añade una
        etiqueta con el valor de su clase.

        save_path es el nombre del directorio en el que se guadarán los
        gráficos.
        """

        if not self._ready_to_save:
            return False

        if indices_matrix is None:
            indices_matrix = [range(len(self._centroids))]

        if self._instances is not None:
            if self._pca is None:
                tmp_instances = []
                for centroid in self._centroids:
                    tmp_instances.append(list(centroid))
                for instance in self._instances:
                    tmp_instances.append(list(instance))
                self._pca = utils.pca_filter(tmp_instances, 2)
            for row in range(indices_matrix):
                plt.figure(row)
                if len(indices_matrix[row]) > 5:
                    title = "Clusters [{} ... {}]".format(indices_matrix[row][0], indices_matrix[row][-1])
                else:
                    title = "Clusters {}".format(indices_matrix[row])
                plt.title(title)
                # iterate on clusters
                for i in range(indices_matrix[row]):
                    c = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]]
                    # cluster centroid
                    plt.scatter(self._pca[i][0], self._pca[i][1], c=c, marker='P')
                    for t in range(len(self._instances)):
                        if self._belonging_bits[t][i]:
                            plt.scatter(self._pca[t + self._k][0], self._pca[t + self._k][1], c=c, marker='.')
                            if tags and self._class_attr is not None:
                                plt.text(self._pca[t + self._k][0], self._pca[t + self._k][1], s=self._data[self._class_attr][t], fontsize=10)
                if save_path_folder is not None:
                    plt.savefig(save_path_folder)
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

    def inter_cluster_dist(self):
        return self._inter_cluster_dist

    def set_inter_cluster_dist(self, inter_cluster_dist):
        self._inter_cluster_dist = inter_cluster_dist

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
    args = _get_args()
    data_path = args.data_path
    output_folder = args.output_folder
    text_attr = args.text_attribute
    class_attr = args.class_attribute
    k = args.k
    m = args.m
    tolerance = args.tolerance
    inter_cluster_distance = args.inter_cluster_distance
    init_strategy = args.init_strategy
    max_it = args.max_it
    verbose = args.verbose
    plot_indices = args.plot_indices
    plot_tags = args.plot_tags
    plot_save_folder = args.plot_save_folder
    KMeans.main(data_path=data_path, output_folder=output_folder, text_attr=text_attr,
                class_attr=class_attr, k=k, tolerance=tolerance, m=m,
                inter_cluster_dist=inter_cluster_distance, init_strat=init_strategy,
                max_it=max_it, plot_indices=plot_indices, plot_tags=plot_tags,
                plot_save_folder=plot_save_folder, verbose=verbose)
