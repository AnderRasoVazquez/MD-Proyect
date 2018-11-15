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
    +{static}INIT_RANDOM = 'random':String
    +{static}INIT_2K = '2k':String
    +{static}TFIDF = 'tfidf':String
    +{static}WEMBEDDINGS = 'word_embeddings':String
    +{static}AVERAGE_LINK = 'average_link':String
    +{static}SINGLE_LINK = 'single_link':String
    +{static}COMPLETE_LINK = 'complete_link':String
  +<<constructor>>Kmeans(output_folder: String, data: String, text_attr: String, class_attr=None: String, k=10: int, tolerance=0.1: Float, m=2: int,
     inter_cluster_dist='average_link': String, w2v_strat='tfidf': String, init_strat='random': String, max_it=50: int)
  +{static}main(output_folder: String, data: String, text_attr: String, class_attr=None: String, k=10: int, tolerance=0.1: Float, m=2: int, inter_cluster_dist='average_link': String,
  {static}init_strat='random': String, max_it=50: int, plot=False: boolean, plot_indices=None, plot_tags=False, plot_save_folder=None: String, verbose=False: boolean): Kmeans

  +form_clusters(verbose=False: boolean)
-_load_instances(w2v_strat:String, attribute:String)
-_initialize_centroids(init_strat: String): Dict
-_init_random_centroids()
-_init_2k()
-_check_finished(): boolean
-_update_belonging_bits()
-_update_centroids()
+save_results(verbose=False: boolean)
+save_clusters(path:String, sorted=True: boolean)
+save_centroids(path:String)
+save_evaluation(path:String)
-_sse(cluster_index: int): Float
-_inter_cluster_distance(inter_cluster_dist: String, first_c: int, second_c: int): Dict
-_single_link(first_c: int, second_c: int): Float
-_complete_link(first_c: int, second_c: int): Float
-_avg_link(first_c: int, second_c: int): Float
+plot(indices_matrix=None: int[][], tags=False: boolean, save_folder=None: String)
+data(): Pandas.Dataframe
+instances(): Numpy.Array
+ k(): int
+set_k(k: int)
+centroids(): Numpy.Array
+bits(): Numpy.Array
+tolerance(): Float
+set_tolerance(tolerance: Float)
+distance_m(): int
+set_distance_m(m: int): int
+inter_cluster_dist(inter_cluster_dist: String): String
+set_inter_cluster_dist()
+max_it(): int
+set_max_it(max_it: int)
+w2v_strat(): String
+set_w2v_strat(w2v_strat: String)
+init_strat(): String
+set_init_stradt(init_strat: String)
+output_folder(): String
+set_output_folder(output_folder: String)
   }


Kmeans -> Distance: <<uses>>
Kmeans -> utils: <<uses>>
Kmeans -> HeatMapPlotter: <<uses>>
Kmeans -> PlotSSE: <<uses>>

```
