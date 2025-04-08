"""This module contains auxiliary functions to be used in the experiments."""

import os
import pandas as pd
import docplex.mp.model as mp
from keras.api import Model as KerasModel


def store_data(file_results: str, dict_results: list[dict]) -> None:
    """Auxiliary function to store the results of the MILP model in a CSV file.
    Args:
        file_results (str): The path to the CSV file.
        dict_results (list[dict]): The list of dictionaries with the results.
    """

    file_exists = os.path.exists(file_results)
    df_new = pd.DataFrame(dict_results)
    df_combined = (
        df_new
        if not file_exists
        else pd.concat([pd.read_csv(file_results), df_new], ignore_index=True)
    )
    df_combined.to_csv(file_results, index=False)


def linear_constraints_to_dict(linear_constraints: list[mp.LinearConstraint]) -> dict:
    """Auxiliary function to get a dictionary from a list of linear constraints.
    Args:
        exp (list[mp.LinearConstraint]): The list of linear constraints.
    Returns:
        dict: A dictionary with the left expression as the key and the right expression as the value.
    """

    return {
        str(e.left_expr): float(e.right_expr.constant)
        for _, e in enumerate(linear_constraints)
    }


def generate_layers_to_relax(
    neural_network: KerasModel, layers_idx: list[int]
) -> list[bool]:
    """Auxiliary function to generate the layers to relax in the MILP model.
    Args:
        neural_network (tf.keras.Model): The neural network model.
        layers_idx (list[int]): The indexes of the layers to relax.
    Returns:
        list[bool]: A list of boolean values indicating which layers to relax.

    Default: All layers not to relax.
    """
    layers_to_relax = [False] * len(neural_network.layers)
    for _, idx in enumerate(layers_idx):
        if not(idx < 0 or idx >= len(neural_network.layers)):
            layers_to_relax[idx] = True
    return layers_to_relax
