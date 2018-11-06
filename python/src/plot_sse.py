"""Utiliza un csv que contiene los errores de los clusters para generar graficos.

usage: plot_sse.py [-h] csv_file output_folder

positional arguments:
  csv_file       Archivo csv a analizar
  output_folder  Carpeta donde guardar los graficos

optional arguments:
  -h, --help     show this help message and exit

example:
python3 plot_sse.py evaluation.csv files/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def _get_args():
    """Devuelve los argumentos introducidos por la terminal."""
    parser = ArgumentParser()
    parser.add_argument('csv_file',
                        type=str,
                        help='Archivo csv a analizar')
    parser.add_argument('output_folder',
                        type=str,
                        help='Carpeta donde guardar los graficos')
    return parser.parse_args()


class PlotSSE(object):
    """Se encarga de hacer graficos de la evaluacion del SSE."""
    
    def __init__(self, csv_file, output_folder):
        self.output_folder = output_folder
        self.df = pd.read_csv(csv_file)
        self._plot_width = 15
        self._plot_height = 3

    def _sse_plot(self):
        """Crea un grafico usando la columna sse del dataframe."""
        df_sse = self.df["sse"].sort_values(ascending=False)
        plt.figure(figsize=(self._plot_width, self._plot_height))
        df_sse.plot("bar")
        plt.title("SSE por cluster")
        output_path_sse = os.path.join(self.output_folder, 'sse_plot.png')
        plt.savefig(output_path_sse)

    def _avg_sse_plot(self):
        """Crea un grafico usando la columna sse_avg del dataframe."""
        df_sse = self.df["sse_avg"].sort_values(ascending=False)
        plt.figure(figsize=(self._plot_width, self._plot_height))
        df_sse.plot("bar")
        plt.title("Media SSE por cluster")
        output_path_sse = os.path.join(self.output_folder, 'sse_avg_plot.png')
        plt.savefig(output_path_sse)

    def create_plots(self):
        """Crea graficos de error a partir de un csv."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self._sse_plot()
        self._avg_sse_plot()


def main():
    args = _get_args()
    plotter = PlotSSE(args.csv_file, args.output_folder)
    plotter.create_plots()


if __name__ == '__main__':
    main()
