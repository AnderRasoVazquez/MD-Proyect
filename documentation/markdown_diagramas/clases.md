# Diagramas de clases

 + plugin atom: `markdown-preview-enchanced`. Para ver la preview click derecho en el editor > Toggle Markdown Preview Enchanced.
 + programa ubuntu: `sudo apt install plantuml`

 "M" indica que es un módulo de Python, no una clase.

```plantuml
class HeatMapPlotter {
  +output_file: String
  +df: Pandas.Dataframe
  +plot_width : int
  +plot_height : int
  +plot_title : String
  +<<constructor>>HeatMapPlotter(csv_file: String, output_file: String,
    plot_width=25 : int, plot_height=10 : int,
    plot_title="MAPA DE CALOR DE ENFERMEDADES POR CLUSTER" : String)
  -_create_dataframe(csv_file: String): Pandas.Dataframe
  -_ordenar_filas(df: Pandas.Dataframe): Pandas.Dataframe
  -_ordenar_columnas(df: Pandas.Dataframe): Pandas.Dataframe
  +crear_grafico()
}

class PlotSSE {
  +output_folder: String
  +df: Pandas.Dataframe
  +plot_width : int
  +plot_height : int
  +plot_title : String
  +<<constructor>>PlotSSE(csv_file: String, output_folder: String,
    plot_width=25 : int, plot_height=10 : int)
  +sse_plot()
  +avg_sse_plot()
  +create_plots()
}


class utils << (M,#FFFFFF) módulo >> {
  +is_number(string: String): boolean
  +tfidf_filter(data_frame: Pandas.Dataframe,
    atributes: int): TfidfMatrix
  +pca_filter(instances: Float[][],
    atributes: int): PCA
}

class Distance {
  -_m: int
  +<<constructor>>Distance(m: int)
  +m(): int
  +set_m(m: int)
  +distance(instance_a: Float,
    instance_b: Float): Float
}

class csv_cleaner << (M,#FFFFFF) módulo >> {
+main()
+clean_string(t: String,
  use_stemmer: boolean): String
}

class Kmeans {
  (en_construccion)
  +<<constructor>>Kmeans()
}


Kmeans -> Distance: uses
Kmeans -> utils: uses

```
