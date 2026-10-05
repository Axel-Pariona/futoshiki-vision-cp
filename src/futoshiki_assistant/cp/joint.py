import math
import time

from ortools.sat.python import cp_model

from ..schema import validate_instance


def probability_to_cost(probability, scale=1000):
    probability = max(float(probability), 1e-9)
    return int(round(-math.log(probability) * scale))


def reify_less_than(model, left, right, name):
    is_less = model.NewBoolVar(name)
    model.Add(left < right).OnlyEnforceIf(is_less)
    model.Add(left >= right).OnlyEnforceIf(is_less.Not())
    return is_less


def build_joint_futoshiki_model(
    data,
    uncertain_relations,
    cost_scale=1000,
):
    validate_instance(data)

    n = data["size"]
    model = cp_model.CpModel()

    grid = [
        [
            model.NewIntVar(1, n, f"x_{row}_{col}")
            for col in range(n)
        ]
        for row in range(n)
    ]

    for row in range(n):
        model.AddAllDifferent(grid[row])

    for col in range(n):
        model.AddAllDifferent(
            [grid[row][col] for row in range(n)]
        )

    for given in data.get("givens", []):
        model.Add(
            grid[given["row"]][given["col"]] == given["value"]
        )

    for relation in data.get("inequalities", []):
        r1, c1 = relation["cell1"]
        r2, c2 = relation["cell2"]

        if relation["operator"] == "<":
            model.Add(grid[r1][c1] < grid[r2][c2])
        else:
            model.Add(grid[r1][c1] > grid[r2][c2])

    decisions = []
    objective_terms = []

    for index, relation in enumerate(uncertain_relations):
        r1, c1 = relation["cell1"]
        r2, c2 = relation["cell2"]
        probabilities = relation["probabilities"]

        left = grid[r1][c1]
        right = grid[r2][c2]

        choose_lt = model.NewBoolVar(f"perception_{index}_lt")
        choose_gt = model.NewBoolVar(f"perception_{index}_gt")
        choose_blank = model.NewBoolVar(f"perception_{index}_blank")

        model.AddExactlyOne([choose_lt, choose_gt, choose_blank])

        model.Add(left < right).OnlyEnforceIf(choose_lt)
        model.Add(left > right).OnlyEnforceIf(choose_gt)

        costs = {
            "<": probability_to_cost(
                probabilities["<"], cost_scale
            ),
            ">": probability_to_cost(
                probabilities[">"], cost_scale
            ),
            "blank": probability_to_cost(
                probabilities["blank"], cost_scale
            ),
        }

        objective_terms.extend(
            [
                costs["<"] * choose_lt,
                costs[">"] * choose_gt,
                costs["blank"] * choose_blank,
            ]
        )

        decisions.append(
            {
                "cell1": [r1, c1],
                "cell2": [r2, c2],
                "variables": {
                    "<": choose_lt,
                    ">": choose_gt,
                    "blank": choose_blank,
                },
                "probabilities": probabilities,
                "costs": costs,
            }
        )

    if objective_terms:
        model.Minimize(sum(objective_terms))

    return model, grid, decisions


def solve_joint_futoshiki(
    data,
    uncertain_relations,
    cost_scale=1000,
):
    model, grid, decisions = build_joint_futoshiki_model(
        data,
        uncertain_relations,
        cost_scale,
    )

    solver = cp_model.CpSolver()

    start = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": solver.StatusName(status),
            "solution": None,
            "selected_relations": None,
            "objective": None,
            "time": elapsed,
        }

    n = data["size"]
    solution = [
        [solver.Value(grid[row][col]) for col in range(n)]
        for row in range(n)
    ]

    selected_relations = []

    for decision in decisions:
        selected = None

        for label, variable in decision["variables"].items():
            if solver.Value(variable):
                selected = label
                break

        selected_relations.append(
            {
                "cell1": decision["cell1"],
                "cell2": decision["cell2"],
                "selected": selected,
                "probabilities": decision["probabilities"],
                "costs": decision["costs"],
            }
        )

    objective = (
        solver.ObjectiveValue() / cost_scale
        if decisions
        else 0.0
    )

    return {
        "status": solver.StatusName(status),
        "solution": solution,
        "selected_relations": selected_relations,
        "objective": objective,
        "time": elapsed,
    }
