import argparse

import numpy as np


def ask_yn(message, prompt='Answer with (y)es or (n)o.'):
    """
    :param message: Question to user.
    :type message: str

    :type prompt: str
    """
    ans = input(f'{message}\n{prompt}').lower()
    if ans == 'y':
        return True
    elif ans == 'n':
        return False
    else:
        print('Unsupported character.')
        return ask_yn(message, prompt)


def flatten_all_but_last(array: np.ndarray):
    return array.reshape((-1, array.shape[-1]))


def one_hot(labels):
    """
    :param labels:
    :type labels: np.ndarray
    :return:
    """
    encoded = np.zeros((labels.shape[0], np.max(labels) + 1), dtype=np.int8)
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded


def one_hot_to_sparse(labels):
    decoded = np.argmax(labels, axis=1)
    return decoded


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def unison_shuffle(a, b, seed=42):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=seed).permutation(len(a))
    return a[p], b[p]


def write_data(x, y, name):
    with open(f'{name}.dat', 'w') as f:
        f.write('x y\n')
        for x_i, y_i in zip(x, y):
            f.write(f'{x_i:.4f} {y_i:.4f}\n')


def write_acc_uc(name=None, **kwargs):
    """

    :param name: Prefix of written files.
    :param kwargs: y values to be written. Keys are used for suffixes.
    """
    for k, v in kwargs.items():
        x = np.arange(len(v))
        write_data(x, v, f'{name}_{k}')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
