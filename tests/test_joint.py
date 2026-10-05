from ortools.sat.python import cp_model

from futoshiki_assistant.cp.joint import reify_less_than


def test_full_reification_less_than():
    model = cp_model.CpModel()

    left = model.NewIntVar(1, 4, "left")
    right = model.NewIntVar(1, 4, "right")

    relation = reify_less_than(
        model,
        left,
        right,
        "is_less",
    )

    model.Add(left == 1)
    model.Add(right == 2)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    )

    assert solver.Value(relation) == 1
