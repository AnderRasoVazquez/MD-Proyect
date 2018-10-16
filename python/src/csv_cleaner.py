import sys
import csv
import string
import nltk
import itertools

from utils import utils
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


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
                # clean last attribute of each row (the string)
                new_row[-1] = clean_string(new_row[-1], True)
                writer.writerow(new_row)


def clean_string(t, use_stemmer):

    #t = string.replace('_x000D_', '').replace('\n', '')
    #t = new_string.replace('/', ' ')

    #sentences = sent_tokenize(t)
    # Genera un array de palabras (word_tokenize)
    tokens = word_tokenize(t)
    # Convierte las palabras en minúsculas(to lower case)
    tokens = [w.lower() for w in tokens]
    # Elimina signos de puntuación
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]


    # Elimina símbolos no alfabéticos
    words = [word for word in stripped if word.isalpha()]

    # Filtra preposiciones (stop-words)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    # Guarda la raíz de las palabras (stemming of words)
    if use_stemmer:
        porter = PorterStemmer()
        words = [porter.stem(word) for word in words]

    # Genera una nueva lista “limpia” de oraciones
    return " ".join(words)


if __name__ == "__main__":
    main()
