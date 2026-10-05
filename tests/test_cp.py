from futoshiki_assistant.cp.solver import (
    check_uniqueness,
    solve_futoshiki,
)


UNIQUE_4X4 = {
    "type": "futoshiki",
    "size": 4,
    "givens": [
        {"row": 0, "col": 0, "value": 1},
        {"row": 0, "col": 1, "value": 2},
        {"row": 0, "col": 2, "value": 3},
        {"row": 0, "col": 3, "value": 4},
        {"row": 1, "col": 0, "value": 2},
        {"row": 1, "col": 1, "value": 3},
        {"row": 1, "col": 2, "value": 4},
        {"row": 1, "col": 3, "value": 1},
        {"row": 2, "col": 0, "value": 3},
        {"row": 2, "col": 1, "value": 4},
        {"row": 2, "col": 2, "value": 1},
        {"row": 2, "col": 3, "value": 2},
        {"row": 3, "col": 0, "value": 4},
        {"row": 3, "col": 1, "value": 1},
        {"row": 3, "col": 2, "value": 2},
        {"row": 3, "col": 3, "value": 3},
    ],
    "inequalities": [],
}


def test_unique_4x4():
    result = check_uniqueness(UNIQUE_4X4)

    assert result["status"] == "UNIQUE"
    assert result["unique"] is True


def test_solve_4x4():
    result = solve_futoshiki(UNIQUE_4X4)

    assert result["solution"] is not None
    assert result["solution"][0] == [1, 2, 3, 4]


def test_multiple_empty_4x4():
    result = check_uniqueness(
        {
            "type": "futoshiki",
            "size": 4,
            "givens": [],
            "inequalities": [],
        }
    )

    assert result["status"] == "MULTIPLE"
