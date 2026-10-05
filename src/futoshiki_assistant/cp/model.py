from ortools.sat.python import cp_model

from ..schema import validate_instance


def build_futoshiki_model(data):
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
        left = grid[r1][c1]
        right = grid[r2][c2]

        if relation["operator"] == "<":
            model.Add(left < right)
        else:
            model.Add(left > right)

    return model, grid
