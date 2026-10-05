import pytest

from futoshiki_assistant.schema import validate_instance


def test_valid_instance():
    data = {
        "type": "futoshiki",
        "size": 4,
        "givens": [
            {"row": 0, "col": 0, "value": 1}
        ],
        "inequalities": [
            {
                "cell1": [0, 0],
                "operator": "<",
                "cell2": [0, 1],
            }
        ],
    }

    assert validate_instance(data) is True


def test_reject_non_adjacent_inequality():
    data = {
        "type": "futoshiki",
        "size": 4,
        "givens": [],
        "inequalities": [
            {
                "cell1": [0, 0],
                "operator": "<",
                "cell2": [0, 2],
            }
        ],
    }

    with pytest.raises(ValueError):
        validate_instance(data)
