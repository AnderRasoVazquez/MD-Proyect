from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import multiprocessing
import gensim.models.word2vec as w2v


def is_number(string):
    """Devuelve True si el string representa un número."""

    try:
        float(string)
        return True
    except ValueError:
        return False


def tfidf_filter(data_frame, attribute):
    """Aplica el filtro TF-IDF a las instancias.

    data_frame: instancias a filtrar. Vienen en forma de dataframe.
    attribute: nombre del atributo texto a transformar."""

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame[attribute].values.astype('U'))
    return tfidf_matrix


def pca_filter(instances, attributes):
    """Aplica el filtro PCA a las instancias.

    instances: instancias a filtrar. Vienen en forma de array de dos dimensiones.
    attributes: número de atributos a reducir los datos.
    """

    pca = PCA(n_components=attributes)
    return pca.fit_transform(instances.copy())


def word_embeddings(data_frame, attribute):
    sentences = []
    for instance_value in data_frame[attribute].get_values().astype('U'):
        if instance_value:
            sentences.append(instance_value.split(' '))

    # Dimensionality of the resulting word vectors.
    #more dimensions, more computationally expensive to train
    #but also more accurate
    #more dimensions = more generalized
    num_features = 300
    # Minimum word count threshold.
    min_word_count = 3
    # Number of threads to run in parallel.
    #more workers, faster we train
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = 7

    # Downsample setting for frequent words.
    #0 - 1e-5 is good for this
    downsampling = 1e-3

    # Seed for the RNG, to make the results reproducible.
    #random number generator
    #deterministic, good for debugging
    seed = 1

    word2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    word2vec.build_vocab(sentences)

    word2vec.train(sentences, epochs=word2vec.iter, total_examples=word2vec.corpus_count)

    #Compress the word vectors into 2D space and plot them
    tsne = TSNE(n_components=2, random_state=0)

    all_word_vectors_matrix = word2vec.wv.syn0

    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[word2vec.wv.vocab[word].index])
                for word in word2vec.wv.vocab
            ]
        ],
        columns=["word", "x", "y"]
    )

    return points.filter(['x', 'y'], axis='columns')
