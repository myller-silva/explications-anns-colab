"""This module contains the functions to generate explanations for the MILP model."""

from typing import Tuple, List
import numpy as np
import docplex.mp.model as mp
from docplex.mp.constr import LinearConstraint


def insert_output_constraints_fischetti(
    mdl, output_variables, network_output, binary_variables
):
    """Insert the output constraints for the MILP model using the method proposed by Fischetti et al."""
    variable_output = output_variables[network_output]
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            p = binary_variables[aux_var]
            aux_var += 1
            mdl.add_indicator(p, variable_output <= output, 1)

    return mdl


def insert_output_constraints_tjeng(
    mdl, output_variables, network_output, binary_variables, output_bounds
):
    """Insert the output constraints for the MILP model using the method proposed by Tjeng et al."""
    variable_output = output_variables[network_output]
    upper_bounds_diffs = (
        output_bounds[network_output][1] - np.array(output_bounds)[:, 0]
    )  # Output i: oi - oj <= u1 = ui - lj
    aux_var = 0

    for i, output in enumerate(output_variables):
        if i != network_output:
            ub = upper_bounds_diffs[i]
            z = binary_variables[aux_var]
            mdl.add_constraint(variable_output - output - ub * (1 - z) <= 0)
            aux_var += 1

    return mdl


def get_minimal_explanation(
    mdl: mp.Model,
    network_input: np.ndarray,
    network_output: int,
    n_classes: int,
    method: str,
    output_bounds: List[Tuple[float, float]] = None,  # type: ignore
    initial_explanation: List[int] = None,  # type: ignore
) -> Tuple[List[LinearConstraint], mp.Model]:
    """Get an explanation for a given input and output using the MILP model."""
    assert not (
        method == "tjeng" and output_bounds is None
    ), "If the method tjeng is chosen, output_bounds must be passed."

    output_variables = [mdl.get_var_by_name(f"o_{i}") for i in range(n_classes)]

    if initial_explanation is None:
        input_constraints = mdl.add_constraints(
            [
                mdl.get_var_by_name(f"x_{i}") == feature.numpy()
                for i, feature in enumerate(network_input[0])
            ],
            names="input",
        )
    else:
        input_constraints = mdl.add_constraints(
            [
                mdl.get_var_by_name(f"x_{i}") == network_input[0][i].numpy()
                for i in initial_explanation
            ],
            names="input",
        )

    binary_variables = mdl.binary_var_list(n_classes - 1, name="b")  # type: ignore
    mdl.add_constraint(mdl.sum(binary_variables) >= 1)  # type: ignore

    if method == "tjeng":
        mdl = insert_output_constraints_tjeng(
            mdl, output_variables, network_output, binary_variables, output_bounds
        )
    else:
        mdl = insert_output_constraints_fischetti(
            mdl, output_variables, network_output, binary_variables
        )

    for constraint in input_constraints:
        mdl.remove_constraint(constraint)
        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)

    inputs = mdl.find_matching_linear_constraints("input")
    return (inputs, mdl)


def get_explanation_relaxed(
    mdl: mp.Model,
    network_input: np.ndarray,
    network_output: int,
    n_classes: int,
    delta: float = 0.1,
) -> Tuple[List[LinearConstraint], mp.Model]:
    """Get an explanation for a given input and output using the relaxed version of the MILP model."""

    output_variables = [mdl.get_var_by_name(f"o_{i}") for i in range(n_classes)]

    input_constraints = mdl.add_constraints(
        [
            mdl.get_var_by_name(f"x_{i}") == feature.numpy()
            for i, feature in enumerate(network_input[0])
        ],
        names="input",
    )

    binary_variables = mdl.binary_var_list(n_classes - 1, name="b")  # type: ignore
    mdl.add_constraint(mdl.sum(binary_variables) >= 1)  # type: ignore

    mdl = insert_output_constraints_fischetti(
        mdl, output_variables, network_output, binary_variables
    )

    x_vars = mdl.find_matching_vars("x_")
    x_values = [feature for i, feature in enumerate(network_input[0])]

    i = 0
    for constraint in input_constraints:
        mdl.remove_constraint(constraint)

        x, v = x_vars[i], float(x_values[i])
        i = i + 1
        left = max(0, v - delta)
        right = min(1, v + delta)

        if left > 0:
            constraint_left = mdl.add_constraint(x >= left)
        if right < 1:
            constraint_right = mdl.add_constraint(x <= right)

        mdl.solve(log_output=False)
        if mdl.solution is not None:
            mdl.add_constraint(constraint)
            if left > 0:
                mdl.remove_constraint(constraint_left)
            if right < 1:
                mdl.remove_constraint(constraint_right)
    inputs = mdl.find_matching_linear_constraints("input")
    return (inputs, mdl)
