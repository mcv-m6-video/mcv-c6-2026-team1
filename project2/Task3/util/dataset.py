from util.io import load_text


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}
