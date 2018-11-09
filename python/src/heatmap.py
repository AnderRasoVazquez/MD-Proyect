""" Utiliza un csv que contiene los clusters para crear un grafico.

usage: heatmap.py [-h] csv_file output_file

positional arguments:
  csv_file     Archivo csv a analizar
  output_file  Nombre del archivo a guardar

optional arguments:
  -h, --help   show this help message and exit
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def _get_args():
    """Devuelve los argumentos introducidos por la terminal."""
    parser = ArgumentParser()
    parser.add_argument('csv_file',
                        type=str,
                        help='Archivo csv a analizar')
    parser.add_argument('output_file',
                        type=str,
                        help='Nombre del archivo a guardar')
    return parser.parse_args()


class HeatMapPlotter(object):
    """Creates a HeatMap."""
    def __init__(self, csv_file, output_file, plot_width=25, plot_height=10, plot_title="MAPA DE CALOR DE ENFERMEDADES POR CLUSTER"):
        self.output_file = output_file
        self.df = self._create_dataframe(csv_file)
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.plot_title = plot_title

    def _create_dataframe(self, csv_file):
        """Crea el dataframe con el formato correcto para hacer un HeatMap."""
        df_base = pd.read_csv(csv_file)

        lista_enfermedades = df_base["gs_text34"].unique()
        lista_clusters = df_base["cluster"].unique()

        df = pd.DataFrame(index=lista_enfermedades, columns=lista_clusters)
        df = df.fillna(0) # rellena el dataframe con ceros

        # rellena el dataframe con valores reales
        for cluster in lista_clusters:
            df_cluster_actual = df_base[df_base["cluster"] == cluster]  # filtrar el df por el cluster actual
            pares = df_cluster_actual["gs_text34"].value_counts()  # {enfermedad: numero_de_casos, ...}
            for enf, times in pares.iteritems():
                df.at[enf, cluster] = times # rellena la columna del cluster correspondiente

        df = self._ordenar_columnas(df)
        df = self._ordenar_filas(df)
        return df

    def _ordenar_filas(self, df):
        """Poner arriba las filas que tengan la enfermedad con el valor más alto."""
        a = dict(df.max(axis=1))
        sorted_by_value = sorted(a.items(), key=lambda kv: kv[1])
        sorted_index = [item[0] for item in sorted_by_value]
        return df.reindex(sorted_index)

    def _ordenar_columnas(self, df):
        """Poner a la izquierda las columnas que tengan la enfermedad con el valor más alto."""
        b = dict(df.max())
        sorted_by_value = sorted(b.items(), key=lambda kv: kv[1], reverse=True)
        sorted_columns = [item[0] for item in sorted_by_value]
        df = df.reindex(columns=sorted_columns)
        return df

    def crear_grafico(self):
        """Guarda el grafico en el archivo indicado."""
        plt.figure(figsize=(self.plot_width, self.plot_height))
        plt.pcolor(self.df, cmap='hot_r', edgecolor='k')
        plt.yticks(np.arange(0.5, len(self.df.index), 1), self.df.index)
        plt.xticks(np.arange(0.5, len(self.df.columns), 1), self.df.columns, rotation=90)
        plt.colorbar()
        plt.title(self.plot_title)

        dir_path = os.path.dirname(os.path.realpath(self.output_file))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Created: " + dir_path)

        plt.savefig(self.output_file)
        print("Created: " + self.output_file)


def main():
    args = _get_args()
    plotter = HeatMapPlotter(args.csv_file, args.output_file)
    plotter.crear_grafico()


if __name__ == '__main__':
    main()
