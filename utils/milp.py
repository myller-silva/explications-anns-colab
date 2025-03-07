"""Mixed Integer Linear Programming (MILP) utilities."""

import numpy as np
import pandas as pd
from cplex import infinity
import docplex.mp.model as mp


def codify_network_relaxed(
    model,
    mdl_original: mp.Model,
    dataframe: pd.DataFrame,
    method: str,
    relaxe_constraints: bool,
    output_bounds_binary_variables,
):
    """Codify a neural network into a MILP model."""

    layers = model.layers
    num_features = layers[0].get_weights()[0].shape[0]
    mdl = mp.Model()

    domain_input, bounds_input = get_domain_and_bounds_inputs(dataframe)
    bounds_input = np.array(bounds_input)

    if relaxe_constraints:
        input_variables = mdl.continuous_var_list(
            num_features, lb=bounds_input[:, 0], ub=bounds_input[:, 1], name="x"  # type: ignore
        )
    else:
        input_variables = []
        for i in range(len(domain_input)):
            lb, ub = bounds_input[i]
            input_variables.append(mdl.continuous_var(lb=lb, ub=ub, name=f"x_{i}"))

    intermediate_variables = []
    auxiliary_variables = []
    decision_variables = []

    for i in range(len(layers) - 1):
        weights = layers[i].get_weights()[0]
        intermediate_variables.append(
            mdl.continuous_var_list(
                weights.shape[1], lb=0, name="y", key_format=f"_{i}_%s"  # type: ignore
            )
        )
        auxiliary_variables.append(
            mdl.continuous_var_list(
                weights.shape[1], lb=0, name="s", key_format=f"_{i}_%s"  # type: ignore
            )
        )
        decision_variables.append(
            mdl.continuous_var_list(
                weights.shape[1], name="a", lb=0, ub=1, key_format=f"_{i}_%s"  # type: ignore
            )
        )

    output_variables = mdl.continuous_var_list(
        layers[-1].get_weights()[0].shape[1], lb=-infinity, name="o"  # type: ignore
    )

    mdl, output_bounds = codify_network_fischetti_relaxed(
        mdl,
        mdl_original,
        layers,
        input_variables,
        auxiliary_variables,
        intermediate_variables,
        decision_variables,
        output_variables,
        output_bounds_binary_variables,
    )

    if relaxe_constraints:
        # Tighten domain of variables 'a'
        for i in decision_variables:
            for a in i:
                a.set_vartype("Continuous")

        # Tighten domain of input variables
        for i, x in enumerate(input_variables):
            if domain_input[i] == "I":
                x.set_vartype("Integer")
            elif domain_input[i] == "B":
                x.set_vartype("Binary")
            elif domain_input[i] == "C":
                x.set_vartype("Continuous")

    return mdl, output_bounds


def codify_network_fischetti_relaxed(
    mdl: mp.Model,
    mdl_original: mp.Model,
    layers: list,
    input_variables: list,
    auxiliary_variables: list,
    intermediate_variables: list,
    decision_variables: list,
    output_variables: list,
    output_bounds_binary_variables: list,
):
    """Codify a neural network into a MILP model using the Fischetti method with relaxed constraints."""
    output_bounds = []

    # for i in range(len(layers)):
    for i, layer in enumerate(layers):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i - 1]
        if i != len(layers) - 1:
            s = auxiliary_variables[i]
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(A.shape[0]):
            if i != len(layers) - 1:
                s_ub = mdl_original.get_var_by_name(f"s_{i}_{j}").ub
                y_ub = mdl_original.get_var_by_name(f"y_{i}_{j}").ub

                m_less = -s_ub  # L
                m_more = y_ub  # U

                y[j].set_lb(max(0, m_less))
                s[j].set_lb(max(0, -m_more))

                y[j].set_ub(max(0, m_more))
                s[j].set_ub(max(0, -m_less))

                if m_more <= 0:
                    mdl.add_constraint(y[j] == 0)
                    continue

                if m_less >= 0:
                    mdl.add_constraint(A[j, :] @ x + b[j] == y[j])
                    continue

                if m_less < 0 and 0 < m_more:
                    mdl.add_constraint(
                        A[j, :] @ x + b[j] == y[j] - s[j], ctname=f"c_{i}_{j}"
                    )
                    mdl.add_constraint(y[j] <= m_more * a[j])
                    mdl.add_constraint(s[j] <= -m_less * (1 - a[j]))
                    continue

            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f"c_{i}_{j}")
                lb, ub = output_bounds_binary_variables[j]
                y[j].set_lb(lb)
                y[j].set_ub(ub)
                output_bounds.append([lb, ub])

    return mdl, output_bounds


def codify_network_fischetti(
    mdl: mp.Model,
    layers: list,
    input_variables: list,
    auxiliary_variables: list,
    intermediate_variables: list,
    decision_variables: list,
    output_variables: list,
):
    "Codify a neural network into a MILP model using the Fischetti method."
    output_bounds = []

    # for i in range(len(layers)):
    for i, layer in enumerate(layers):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i - 1]
        if i != len(layers) - 1:
            s = auxiliary_variables[i]
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(A.shape[0]):

            if i != len(layers) - 1:
                mdl.add_constraint(
                    A[j, :] @ x + b[j] == y[j] - s[j], ctname=f"c_{i}_{j}"
                )
                mdl.add_indicator(a[j], y[j] <= 0, 1)
                mdl.add_indicator(a[j], s[j] <= 0, 0)

                mdl.maximize(y[j])
                mdl.solve()
                ub_y = mdl.solution.get_objective_value() # type: ignore
                mdl.remove_objective()

                mdl.maximize(s[j])
                mdl.solve()
                ub_s = mdl.solution.get_objective_value() # type: ignore
                mdl.remove_objective()

                y[j].set_ub(ub_y)
                s[j].set_ub(ub_s)

            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f"c_{i}_{j}")
                mdl.maximize(y[j])
                mdl.solve()
                ub = mdl.solution.get_objective_value() # type: ignore
                mdl.remove_objective()

                mdl.minimize(y[j])
                mdl.solve()
                lb = mdl.solution.get_objective_value() # type: ignore
                mdl.remove_objective()

                y[j].set_ub(ub)
                y[j].set_lb(lb)
                output_bounds.append([lb, ub])

    return mdl, output_bounds


def codify_network_tjeng(
    mdl: mp.Model,
    layers: list,
    input_variables: list,
    intermediate_variables: list,
    decision_variables: list,
    output_variables: list,
):
    output_bounds = []

    for i in range(len(layers)):
        A = layers[i].get_weights()[0].T
        b = layers[i].bias.numpy()
        x = input_variables if i == 0 else intermediate_variables[i - 1]
        if i != len(layers) - 1:
            a = decision_variables[i]
            y = intermediate_variables[i]
        else:
            y = output_variables

        for j in range(A.shape[0]):

            mdl.maximize(A[j, :] @ x + b[j])
            mdl.solve()
            ub = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if ub <= 0 and i != len(layers) - 1:
                print("ENTROU, o ub é negativo, logo y = 0")
                mdl.add_constraint(y[j] == 0, ctname=f"c_{i}_{j}")
                continue

            mdl.minimize(A[j, :] @ x + b[j])
            mdl.solve()
            lb = mdl.solution.get_objective_value()
            mdl.remove_objective()

            if lb >= 0 and i != len(layers) - 1:
                print("ENTROU, o lb >= 0, logo y = Wx + b")
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j], ctname=f"c_{i}_{j}")
                continue

            if i != len(layers) - 1:
                mdl.add_constraint(y[j] <= A[j, :] @ x + b[j] - lb * (1 - a[j]))
                mdl.add_constraint(y[j] >= A[j, :] @ x + b[j])
                mdl.add_constraint(y[j] <= ub * a[j])

                # mdl.maximize(y[j])
                # mdl.solve()
                # ub_y = mdl.solution.get_objective_value()
                # mdl.remove_objective()
                # y[j].set_ub(ub_y)

            else:
                mdl.add_constraint(A[j, :] @ x + b[j] == y[j])
                # y[j].set_ub(ub)
                # y[j].set_lb(lb)
                output_bounds.append([lb, ub])

    return mdl, output_bounds


def codify_network(
    model: object, dataframe: pd.DataFrame, method: str, relaxe_constraints: bool
):
    """Codify a neural network into a MILP model."""
    layers = model.layers # type: ignore
    num_features = layers[0].get_weights()[0].shape[0]
    mdl = mp.Model()

    domain_input, bounds_input = get_domain_and_bounds_inputs(dataframe)
    bounds_input = np.array(bounds_input)

    if relaxe_constraints:
        input_variables = mdl.continuous_var_list(num_features, lb=bounds_input[:, 0], ub=bounds_input[:, 1], name="x")  # type: ignore
    else:
        input_variables = []
        for i in range(len(domain_input)):
            lb, ub = bounds_input[i]
            if domain_input[i] == "C":
                input_variables.append(mdl.continuous_var(lb=lb, ub=ub, name=f"x_{i}"))
            elif domain_input[i] == "I":
                input_variables.append(mdl.integer_var(lb=lb, ub=ub, name=f"x_{i}"))
            elif domain_input[i] == "B":
                input_variables.append(mdl.binary_var(name=f"x_{i}"))

    intermediate_variables = []
    auxiliary_variables = []
    decision_variables = []

    for i in range(len(layers) - 1):
        weights = layers[i].get_weights()[0]
        intermediate_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name="y", key_format=f"_{i}_%s"))  # type: ignore

        if method == "fischetti":
            auxiliary_variables.append(mdl.continuous_var_list(weights.shape[1], lb=0, name="s", key_format=f"_{i}_%s"))  # type: ignore

        if relaxe_constraints and method == "tjeng":
            decision_variables.append(mdl.continuous_var_list(weights.shape[1], name="a", lb=0, ub=1, key_format=f"_{i}_%s"))  # type: ignore
        else:
            decision_variables.append(mdl.binary_var_list(weights.shape[1], name="a", lb=0, ub=1, key_format=f"_{i}_%s"))  # type: ignore

    output_variables = mdl.continuous_var_list(layers[-1].get_weights()[0].shape[1], lb=-infinity, name="o")  # type: ignore

    if method == "tjeng":
        mdl, output_bounds = codify_network_tjeng(
            mdl,
            layers,
            input_variables,
            intermediate_variables,
            decision_variables,
            output_variables,
        )
    else:
        mdl, output_bounds = codify_network_fischetti(
            mdl,
            layers,
            input_variables,
            auxiliary_variables,
            intermediate_variables,
            decision_variables,
            output_variables,
        )

    if relaxe_constraints:
        # Tighten domain of variables 'a'
        for i in decision_variables:
            for a in i:
                a.set_vartype("Integer")

        # Tighten domain of input variables
        for i, x in enumerate(input_variables):
            if domain_input[i] == "I":
                x.set_vartype("Integer")
            elif domain_input[i] == "B":
                x.set_vartype("Binary")
            elif domain_input[i] == "C":
                x.set_vartype("Continuous")

    return mdl, output_bounds


def get_domain_and_bounds_inputs(dataframe: pd.DataFrame):
    domain = []
    bounds = []
    for column in dataframe.columns[:-1]:
        if len(dataframe[column].unique()) == 2:
            domain.append("B")
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        elif np.any(
            dataframe[column].unique().astype(np.int64)
            != dataframe[column].unique().astype(np.float64)
        ):
            domain.append("C")
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])
        else:
            domain.append("I")
            bound_inf = dataframe[column].min()
            bound_sup = dataframe[column].max()
            bounds.append([bound_inf, bound_sup])

    return domain, bounds
