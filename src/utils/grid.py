from itertools import groupby, product
from typing import Mapping

import collections


def linearize(dictionary: Mapping):
    """
    Linearize a nested dictionary making keys, tuples
    :param dictionary: nested dict
    :return: one level dict
    """
    exps = []
    for key, value in dictionary.items():
        if isinstance(value, collections.abc.Mapping):
            exps.extend(((key, lin_key), lin_value) for lin_key, lin_value in linearize(value))
        elif isinstance(value, list):
            exps.append((key, value))
        else:
            raise ValueError("Only dict or lists!!!")
    return exps


def extract(elem: tuple):
    """
    Exctract the element of a single element tuple
    :param elem: tuple
    :return: element of the tuple if singleton or the tuple itself
    """
    if len(elem) == 1:
        return elem[0]
    return elem


def delinearize(lin_dict):
    """
    Convert a dictionary where tuples can be keys in na nested dictionary
    :param lin_dict: dicionary where keys can be tuples
    :return:
    """
    # Take keys that are tuples
    filtered = list(filter(lambda x: isinstance(x[0], tuple), lin_dict.items()))
    # Group it to make one level
    grouped = groupby(filtered, lambda x: x[0][0])
    # Create the new dict and apply recursively
    new_dict = {k: delinearize({extract(elem[0][1:]): elem[1] for elem in v}) for k, v in grouped}
    # Remove old items and put new ones
    for key, value in filtered:
        lin_dict.pop(key)
    delin_dict = {**lin_dict, **new_dict}
    return delin_dict


def make_grid(dict_of_list):
    """
    Produce a list of dict for each combination of values in the input dict given by the list of values
    :param dict_of_list: a dictionary where values can be lists
    :return: a list of dictionaries given by the cartesian product of values in the input dictionary
    """
    # Linearize the dict to make the cartesian product straight forward
    linearized_dict = linearize(dict_of_list)
    # Compute the grid
    keys, values = zip(*linearized_dict)
    grid_dict = list(dict(zip(keys, values_list)) for values_list in product(*values))
    # Delinearize the list of dicts
    return [delinearize(dictionary) for dictionary in grid_dict]
