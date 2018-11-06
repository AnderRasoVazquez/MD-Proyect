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

    # ordenadas de mayor a menor número de apariciones
    strings_to_ommit = ['',
                        'client had no additional point',
                        'the client thanked for the service',
                        'respondent had nothing to add',
                        'participant had nothing to add',
                        'mother had nothing to add',  # podría ser interesante por lo de la madre
                        'the client has nothing to suggest',
                        'the participant has nothing to add',
                        'respondent thanked for being visited',
                        'participant had no addition point',
                        'respondent said that the service was good',
                        'respondent explained that the cause of death was fall and hurt the back and failed to get treatment',
                        'the client had nothing to suggest',
                        'the client had nothing to add',
                        'respondent thanked for the service received',
                        'participant has nothing to add',
                        'respondent thanked for the service',
                        'the client thanked for services which provided by nurses and doctors in the hospital especial [HOSPITAL]',
                        'no comments.',
                        'mother had nothing to add, the service at the hospital was good',
                        'participant had no additional point',
                        'client had nothing to add',
                        'the client thanked for services which provided by nurses and doctors in the hospital_x000D__x000D_\n the client misplaced death certificate',
                        'the client thanked for service',
                        'no further comments',
                        'the client has nothing to add',
                        'participant thanked for service',
                        'respondent was satisfied with the service',
                        'the client thanked for services_x000D__x000D_\n the client misplaced death certificate',
                        'mother had nothing to add, satisfied with the service',
                        'participant thanked for services',
                        'the client thanked for service which provided by nurse and doctor in the hospital especial [HOSPITAL]',
                        'respondent was satisfied with the service received',
                        'service was good',
                        'the client thanked for service which provided by nurse and doctor in the hospitali especial [HOSPITAL]',
                        'client thanked for services',
                        'respondent said that the service was not good',
                        'the client thanked for service the client misplaced death certificate',
                        'client said service was good',
                        'client thanked for the service',
                        'no further commentsno comments.',
                        'client thanks for the service',
                        'mother satisfied with hospital service',
                        'she said service was good',
                        'the client misplaced death certificate the client thanked for service',
                        'the interviewee did not want to add anything morethe interview went smoothly.',
                        'client had  no additional point',
                        'client had no additional point service provided was good',
                        'client had nothing to suggest',
                        'respondent said that the service was not good ',
                        'respondent thanked for the service, it was good',
                        'the client thanked  for the service',
                        'the client thanked for the service.',
                        'the participant had nothing to add',
                        'the participant thanked for the service',
                        'client  had no additional point',
                        'client commended on good service provided',
                        'client thank for the service',
                        'client thanked for service',
                        'client thanked for services which provided at [HOSPITAL]',
                        'mother had no additional point',
                        'participant thanked for service which provided by nurse doctor especially [HOSPITAL]',
                        'participant thanked for services which provided by nurses and doctors especially [HOSPITAL]',
                        'respondent said that the service was normal.',
                        'respondent was satisfied with the service received.  ',
                        'the client said that the service was good',
                        'the client suggest nothing the client misplaced death certificate',
                        'the client thanked for services which provided by nurses and doctors in the hospital',
                        'the client thanked for services which provided by nurses and doctors in the hospital especial [HOSPITAL]_x000D__x000D_\n the client misplaced death certificate',
                        'the client thanked for the services',
                        'the client thanked for the services they got at [HOSPITAL]',
                        'did not want to give any more informationthe interview went smoothly.',
                        'mother commended that she was well attended',
                        'mother had nothing to add.',
                        'no comment',
                        'no commentno comment',
                        'no comments',
                        'no comments.no comments.',
                        'nothing to say',
                        'participant thanked for service especially [HOSPITAL]',
                        'participant thanked for service which provided by nurse especially [HOSPITAL]',
                        'participant thanked for services which provided by nurses and doctors',
                        'participant thanked for the services which provided by nurses and doctors especially [HOSPITAL]',
                        'respondent thanked  for the service received.',
                        'the client had nothing to comment',
                        'the client had nothing to say',
                        'the client misplaced death certificate_x000D__x000D_\n the client thanked for services which provided by nurses and doctors in the hospital',
                        'the client thanked for service which provided by nurse and doctor in the hospital the client misplaced death certificate',
                        'the client thanked for service which provided by nurse doctor in the hospital the client misplaced death certificate',
                        'the client was satisfied with the service',
                        '[PERSON] didn\'t want to add anything else.the interview went smoothly.',
                        'client had no addition point',
                        'client had no additional point service was good',
                        'client has no additional point',
                        'client has nothing to suggest',
                        'client said service was poor',
                        'he ',
                        'didn\'t know the cause of death',
                        'mother had nothing to add, she satisfied with the service',
                        'mother satisfied with the service',
                        'no comment.the interview was fluent',
                        'no commentsthe interview went smoothly.',
                        'no further comments.the interview took place inside the home with no problem.',
                        'no further commentsthe interview flowed smoothly.',
                        'no further commentsthe interview was fluent',
                        'no.smooth interview. the informant was very nice.',
                        'nonenone',
                        'nono',
                        'participant  had nothing to add',
                        'participant had no addition  point',
                        'participant had no attional point',
                        'participant had nothing to add.',
                        'participant thanked for services which provided by [HOSPITAL]',
                        'participant thanked for the service',
                        'participant thanked very much for services which provided by nurses and doctors especially [HOSPITAL]',
                        'respondent didn\'t have death certificate, was sent to home of origin',
                        'respondent didn\'t know the cause of death',
                        'respondent had no additional point, services were good.',
                        'respondent had nothing to add or complains',
                        'respondent had nothing to add, service was good',
                        'respondent said that she received good service.',
                        'respondent said that the service was bad',
                        'respondent thanked  that the service was good',
                        'respondent thanked for being visited, hospital service was not good',
                        'respondent thanked for being visited, hospital service was not good',
                        'respondent thanked for good service.',
                        'respondent was satisfied with the service at [HOSPITAL]',
                        'the client claims about the service which provided by nurse and doctor in the hospital especial [HOSPITAL]',
                        'the client claims about the services which provided by nurses and doctors in the hospital especial [HOSPITAL]_x000D__x000D_\n the client misplaced death certificate',
                        'the client had no additional information',
                        'the client had no comment',
                        'the client had no suggestion',
                        'the client had nothing to add_x000D__x000D_\n the client misplaced death certificate',
                        'the client has nothing to comment',
                        'the client is pleased with the medical services provided at [HOSPITAL]',
                        'the client misplaced death certificate_x000D__x000D_\n the client thanked for services',
                        'the client nothing to add',
                        'the client nothing to add',
                        'the client said that the service was not satisfactory',
                        'the client said that the service was very poor',
                        'the client thanked for service the client misplaced  death certificate',
                        'the client thanked for service the client transfer death certificate to their original home [PLACE]',
                        'the client thanked for service which provided at [HOSPITAL]',
                        'the client thanked for service which provided by nurse and doctor in the hospitali',
                        'the client thanked for service_x000D__x000D_\n the client misplaced death certificate',
                        'the client thanked for service,the client misplaced death certificate',
                        'the client thanked for services which provided by nurses and doctors in the hospital especial [HOSPITAL].',
                        'the client thanked for services which provided by nurses and doctors in the hospital especial [HOSPITAL]._x000D__x000D_\n the client misplaced death certificate',
                        'the client thanked for services.',
                        'the client thanked for the service the death certificate has been misplaced',
                        'the informant did not want to add anything else.the interview went smoothly.',
                        'the mother had no additional point',
                        'the participant satisfied for the medical service provided at [HOSPITAL]',
                        'the participant satisfied for the service provided at [HOSPITAL]',
                        'the participant thanked for service which provided by nurse especially [HOSPITAL]',
                        'the participant thanked for services',
                        'the participant thanked for services which provided at [HOSPITAL]',
                        'the participant thanked for the services',
                        'the service was good',
                        'they didn\'t want to add anything else.the interview went smoothly.'
                        ]

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
                if not new_row[-1] in strings_to_ommit:
                    new_row[-1] = clean_string(new_row[-1], True)
                    writer.writerow(new_row)


def clean_string(t, use_stemmer):
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
