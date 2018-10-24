# for reference in color codes: https://github.com/DavidPerezGomez/ANSI-escape-codes/blob/master/ANSI%20escapes
import sys
import os
import numpy as np
import pandas as pn


def get_exec_path():
    return os.path.abspath(os.path.join(sys.path[0], os.pardir, os.pardir, 'run'))


def get_files_path():
    return os.path.abspath(os.path.join(sys.path[0], os.pardir, os.pardir, 'files'))


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def csv_to_sparse_csv(path1, path2):
    file1 = open(path1, 'w')
    file2 = open(path2, 'r')
    file1.write(file2.readline())
    for index, line in enumerate(file2):
        v = [float(x) if is_number(x) else x for x in line.split(',')]
        file1.write(vector_to_sparse_string(v, 0) + '\n')
    file1.close()
    file2.close()


def sparse_csv_to_dataframe(path, empty_value):
    with open(path, 'r') as f:
        for index, line in enumerate(f):
            pass
        line_count = index + 1
    with open(path, 'r') as f:
        dict = {}
        headers = f.readline().split(',')
        for header in headers:
            if header.endswith('\n'):
                header = header[:-1]
            if header.startswith('\''):
                header = header[1:-1]
            dict[header] = [empty_value] * (line_count - 1)
        for instance_index, line in enumerate(f):
            instance_index -= 1
            for element in line.split(','):
                attr_index, value = element.split(' ', maxsplit=1)
                attr_index = int(attr_index)
                if is_number(value):
                    value = float(value)
                elif value.startswith('\''):
                    value = value[1:-1]
                dict[headers[attr_index]][instance_index] = value
    return pn.DataFrame(dict).to_sparse(empty_value)


def dataframe_to_sparse_csv(data, path, empty_value):
    with open(path, 'w') as f:
        f.write('{}\n'.format(','.join([str(x) for x in data.columns])))
        for value in data.get_values():
            f.write('{}\n'.format(vector_to_sparse_string(value, empty_value)))


def sparse_string_to_vector(string, v_length, empty_value):
    vector = np.array([empty_value] * v_length, dtype='object')
    for element in string.split(','):
        index, value = element.split(' ')
        index = int(index)
        if is_number(value):
            value = float(value)
        vector[int(index)] = value
    return vector


def vector_to_sparse_string(vector, value_to_omit):
    string = ""
    for i in range(len(vector)):
        if vector[i] != value_to_omit:
            string += '{} {},'.format(i, vector[i])
    return string[:-1]


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
