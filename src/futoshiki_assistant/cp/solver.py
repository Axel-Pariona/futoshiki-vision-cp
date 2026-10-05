import time

from ortools.sat.python import cp_model

from .model import build_futoshiki_model


class UniquenessChecker(cp_model.CpSolverSolutionCallback):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.solution_count = 0
        self.first_solution = None

    def on_solution_callback(self):
        self.solution_count += 1

        if self.solution_count == 1:
            self.first_solution = [
                [self.Value(variable) for variable in row]
                for row in self.grid
            ]

        if self.solution_count >= 2:
            self.StopSearch()


def solve_futoshiki(data):
    model, grid = build_futoshiki_model(data)
    solver = cp_model.CpSolver()

    start = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - start

    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return {
            "status": solver.StatusName(status),
            "solution": None,
            "solve_time": elapsed,
        }

    n = data["size"]
    solution = [
        [solver.Value(grid[row][col]) for col in range(n)]
        for row in range(n)
    ]

    return {
        "status": solver.StatusName(status),
        "solution": solution,
        "solve_time": elapsed,
    }


def check_uniqueness(data):
    model, grid = build_futoshiki_model(data)
    solver = cp_model.CpSolver()
    callback = UniquenessChecker(grid)

    start = time.perf_counter()
    solver.SearchForAllSolutions(model, callback)
    elapsed = time.perf_counter() - start

    if callback.solution_count == 0:
        return {
            "status": "INFEASIBLE",
            "unique": False,
            "solutions_found": 0,
            "time": elapsed,
            "solution": None,
        }

    if callback.solution_count == 1:
        return {
            "status": "UNIQUE",
            "unique": True,
            "solutions_found": 1,
            "time": elapsed,
            "solution": callback.first_solution,
        }

    return {
        "status": "MULTIPLE",
        "unique": False,
        "solutions_found": 2,
        "time": elapsed,
        "solution": callback.first_solution,
    }
