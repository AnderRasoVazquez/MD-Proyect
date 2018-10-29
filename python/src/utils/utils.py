# for reference in color codes: https://github.com/DavidPerezGomez/ANSI-escape-codes/blob/master/ANSI%20escapes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def tfidf_filter(data_frame, attribute):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame[attribute].values.astype('U'))
    return tfidf_matrix


def pca_filter(instances, attributes):
    pca = PCA(n_components=attributes)
    return pca.fit_transform(instances.copy())


# def csv_to_sparse_csv(path1, path2):
#     file1 = open(path1, 'w')
#     file2 = open(path2, 'r')
#     file1.write(file2.readline())
#     for index, line in enumerate(file2):
#         v = [float(x) if is_number(x) else x for x in line.split(',')]
#         file1.write(vector_to_sparse_string(v, 0) + '\n')
#     file1.close()
#     file2.close()
#
#
# def sparse_csv_to_dataframe(path, empty_value):
#     with open(path, 'r') as f:
#         vector = []
#         headers = f.readline().split(',')
#         for i in range(len(headers)):
#             header = headers[i]
#             if header.endswith('\n'):
#                 header = header[:-1]
#                 headers[i] = header
#             if header.startswith('\''):
#                 header = header[1:-1]
#                 headers[i] = header
#         data = []
#         for index, line in enumerate(f):
#             vector = [empty_value] * len(headers)
#             for element in line.split(','):
#                 attr_index, value = element.split(' ', maxsplit=1)
#                 attr_index = int(attr_index)
#                 if is_number(value):
#                     value = float(value)
#                 elif value.startswith('\''):
#                     value = value[1:-1]
#                 vector[attr_index] = value
#             data.append(vector)
#     return pn.SparseDataFrame(data, columns=headers)
#
#
# def dataframe_to_sparse_csv(data, path, empty_value):
#     with open(path, 'w') as f:
#         f.write('{}\n'.format(','.join([str(x) for x in data.columns])))
#         for value in data.get_values():
#             f.write('{}\n'.format(vector_to_sparse_string(value, empty_value)))
#
#
# def sparse_string_to_vector(string, v_length, empty_value):
#     vector = np.array([empty_value] * v_length, dtype='object')
#     for element in string.split(','):
#         index, value = element.split(' ')
#         index = int(index)
#         if is_number(value):
#             value = float(value)
#         vector[int(index)] = value
#     return vector
#
#
# def vector_to_sparse_string(vector, value_to_omit):
#     string = ""
#     for i in range(len(vector)):
#         if vector[i] != value_to_omit:
#             string += '{} {},'.format(i, vector[i])
#     return string[:-1]


def turn_color(color, text):
    return '\033[{}m{}\033[0m'.format(color, text)


def turn_green(text):
    return turn_color(92, text)


def turn_yellow(text):
    return turn_color(33, text)


def turn_red(text):
    return turn_color(91, text)


def print_warning(text):
    print(turn_yellow(text))


def print_error(text):
    print(turn_red(text))


# class SparseVector:
#
#     def __init__(self, indices=[], values=[]):
#         self._indices = indices.copy()
#         self._indices.sort()
#         self._values = {}
#         for i in range(len(self._indices)):
#             self._values[indices[i]] = values[i]
#
#     def get_indices(self):
#         return self._indices.copy()
#
#     def get_values(self):
#         return self._values.copy()
#
#     def add(self, index, value):
#         if value != 0:
#             if index in self._indices:
#                 self._values[index] += value
#                 if self._values[index] == 0:
#                     self._values.pop(index, None)
#                     self._indices.remove(index)
#             else:
#                 self._values[index] = value
#                 bisect.insort(self._indices, index)
#
#     def copy(self):
#         copy = SparseVector()
#         copy._indices = self._indices.copy()
#         copy._values = self._values.copy()
#         return copy
#
#     def __len__(self):
#         return len(self._indices)
#
#     def __add__(self, other):
#         return self._operate(other, True)
#
#     def __sub__(self, other):
#         return self._operate(other, False)
#
#     def __str__(self):
#         string = '['
#         for index in self._indices:
#             string += '({}, {}), '.format(index, self._values[index])
#         string = string[:-2] + ']'
#         return string
#
#     def _operate(self, other, addition):
#         i = 0
#         j = 0
#         sum_indices = []
#         sum_values = []
#         while i < len(self) and j < len(other):
#             index_a = self._indices[i]
#             index_b = other.get_indices()[j]
#             if index_a < index_b:
#                 sum_indices.append(index_a)
#                 sum_values.append(self._values[index_a])
#                 i += 1
#             elif index_b < index_a:
#                 sum_indices.append(index_b)
#                 sum_values.append(other.get_values()[index_b])
#                 j += 1
#             else:
#                 sum_indices.append(index_a)
#                 if addition:
#                     sum_values.append(self._values[index_a] + other.get_values()[index_b])
#                 else:
#                     sum_values.append(self._values[index_a] - other.get_values()[index_b])
#                 i += 1
#                 j += 1
#         return SparseVector(sum_indices, sum_values)
