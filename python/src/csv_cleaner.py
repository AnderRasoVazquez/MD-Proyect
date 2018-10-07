import sys
import csv
from utils import utils


def main():
    try:
        file = sys.argv[1]
        new_file = sys.argv[2]
    except IndexError:
        documentation = "Reformatea un archivo .cvs para eliminar los saltos de linea en campos de tipo string en la " \
                        "última posición que puedan causar problemas.\n" \
                        "Dos argumentos esperados:\n" \
                        "\t1 - Ruta del archivo .csv que se quiere limpiar.\n" \
                        "\t2 - Ruta del archivo .csv en el que guardar el resultado.\n" \
                        "Ejemplo: python csv_cleaner.py file.csv file_clean.csv"
        utils.print_warning(documentation)
        sys.exit(1)

    with open(file, 'r', newline='') as f:
        # create the csv reader for the input file
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        with open(new_file, 'w', newline='') as nf:
            # create the csv writer for the output file
            writer = csv.writer(nf, delimiter=',', quotechar='\"', quoting=csv.QUOTE_ALL)
            # first line is the header row. always take it
            writer.writerow(reader.__next__())
            # the first has already been read, so it isn't in 'reader' anymore
            # iterate on remaining lines
            for row in reader:
                new_row = row.copy()
                # delete all '\n' (line breaks) from the last element of the row
                new_row[-1] = new_row[-1].replace('_x000D_', '').replace('\n', '')
                writer.writerow(new_row)


if __name__ == "__main__":
    main()
