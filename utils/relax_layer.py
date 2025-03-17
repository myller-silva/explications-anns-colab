"""Relax a layer of a neural network."""

import docplex.mp.model as mp


def relax_layer(*args, **kwargs) -> None:
    """Relax a layer of a neural network.
    Args:
    i: Index of the layer.
    j: Index of the neuron.

    Kwargs:
    model: The MILP model.

    Returns:
    None.
    """

    i, j = args
    model: mp.Model = kwargs["model"]
    s, y, a = kwargs["s"], kwargs["y"], kwargs["a"]
    x = kwargs["x"]
    A, b = kwargs["A"], kwargs["b"]

    # Remove constraints for the neuron
    remove_layer_constraints(i, j, model=model)

    # Change the type of the variable to continuous
    a[j].set_vartype("Continuous")

    # Add the new constraints for the relaxed layer
    add_layer_constraints(A, b, x, y, s, a, i, j, model=model)


def remove_layer_constraints(*args, **kwargs) -> None:
    """Remove constraints for a layer of a neural network.
    Args:
    i: Index of the layer.
    j: Index of the neuron.

    Kwargs:
    model: The MILP model.

    Returns:
    None.
    """

    i, j = args
    model: mp.Model = kwargs["model"]

    model.remove_constraints(
        [
            model.get_constraint_by_name(f"c_{i}_{j}"),
            model.get_constraint_by_name(f"c_s_{i}_{j}"),
            model.get_constraint_by_name(f"c_y_{i}_{j}"),
        ]
    )


def add_layer_constraints(*args, **kwargs) -> None:
    """Create other constraints for a layer of a neural network.
    Args:
    A: Weights of the layer.
    b: Bias of the layer.
    x: Input variables.
    y: Output variables.
    s: Auxiliary variables.
    a: Decision variables.
    i: Index of the layer.
    j: Index of the neuron.

    Kwargs:
    model: The MILP model.

    Returns:
    None.
    """

    A, b, x, y, s, a, i, j = args
    model: mp.Model = kwargs["model"]
    m_less, m_more = -s[j].ub, y[j].ub  # L, U

    if m_more <= 0:
        model.add_constraint(y[j] == 0, ctname=f"c_{i}_{j}")
        return

    if m_less >= 0:
        model.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f"c_{i}_{j}")
        return

    if m_less < 0 < m_more:
        model.add_constraints(
            [
                A[j, :] @ x + b[j] == y[j] - s[j],
                y[j] <= m_more * a[j],
                s[j] <= -m_less * (1 - a[j]),
            ],
            names=[f"c_{i}_{j}", f"c_y_{i}_{j}", f"c_s_{i}_{j}"],
        )
        return
