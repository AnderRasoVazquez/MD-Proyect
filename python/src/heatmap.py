import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        file = sys.argv[1]
        path_to_save = sys.argv[2]
    except IndexError:
        documentation = "Genera un heatmap dadp un archivo .csv en el que se disponga del cluster asignado a cada una" \
                        " de las instancias. Dos argumentos esperado:\n" \
                        "\t1 - Ruta del archivo .csv con la información de los cluster.\n" \
                        "\t2 - Ruta donde guardar el heatmap en formato .png.\n" \
                        "Ejemplo: python heatmap.py clustered_instances.csv /path/to/heatmap.png"
        print(documentation)
        sys.exit(1)

    clusters = []
    death = []
    clusters_dict = {}
    death_dict = {}

    with open(file, 'r', newline='') as f:
        # Create the csv reader for the input file
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        # First line is the header row. Check the index of the cluster and the cause of death.
        first_row = reader.__next__()
        clusters_i = first_row.index('cluster')
        death_i = first_row.index('gs_text34')
        # Iterate on the remaining lines to get all posible clusters and causes of death.
        # Also stores in a dictionary how many instances are in every cluster and cause of death.
        for row in reader:
            if not row[clusters_i] in clusters:
                clusters.append(row[clusters_i])
                clusters_dict[row[clusters_i]] = 1
            else:
                clusters_dict[row[clusters_i]] += 1
            if not row[death_i] in death:
                death.append(row[death_i])
                death_dict[row[death_i]] = 1
            else:
                death_dict[row[death_i]] += 1

        # Sorts the causes of death by their number of instaces.
        sorted_death = []
        for a, b in sorted(death_dict.items(), key=lambda x: x[1], reverse=False):
            sorted_death.append(a)

        # Sorts the clusters by their number of instaces.
        sorted_clusters = []
        for a, b in sorted(clusters_dict.items(), key=lambda x: x[1], reverse=True):
            sorted_clusters.append(a)

        # Creates a zero filled 2D array with dimensions of number of causes of death and number of clusters.
        heatmap = np.zeros([len(sorted_death), len(sorted_clusters), ])
        # Resets the reader to the start and skips the header row.
        f.seek(0)
        reader.__next__()
        # Fills the 2D array with the number of appearances of each cause of death in each cluster
        for row in reader:
            heatmap[sorted_death.index(row[death_i]), sorted_clusters.index(row[clusters_i])] += 1
        f.close()

    fig, ax = plt.subplots()
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    ax.imshow(heatmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(sorted_clusters)))
    ax.set_yticks(np.arange(len(sorted_death)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(sorted_clusters)
    ax.set_yticklabels(sorted_death)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    ax.set_title("Heatmap")
    fig.tight_layout()
    plt.imshow(heatmap, origin="lower", cmap='hot_r', interpolation='nearest')
    plt.colorbar()
    plt.savefig(path_to_save, dpi=2000)
    plt.show()


if __name__ == "__main__":
    main()
