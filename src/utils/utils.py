# for reference in color codes: https://github.com/DavidPerezGomez/ANSI-escape-codes/blob/master/ANSI%20escapes


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
