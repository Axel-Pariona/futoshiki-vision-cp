def validate_instance(data):
    if not isinstance(data, dict):
        raise ValueError("La instancia debe ser un diccionario.")

    if data.get("type") != "futoshiki":
        raise ValueError("type debe ser 'futoshiki'.")

    n = data.get("size")
    if type(n) is not int or n < 2:
        raise ValueError("size debe ser un entero mayor o igual a 2.")

    seen_givens = {}

    for given in data.get("givens", []):
        if not isinstance(given, dict):
            raise ValueError("Cada given debe ser un diccionario.")

        r = given.get("row")
        c = given.get("col")
        value = given.get("value")

        if not all(type(x) is int for x in (r, c, value)):
            raise ValueError("row, col y value deben ser enteros.")

        if not (0 <= r < n and 0 <= c < n and 1 <= value <= n):
            raise ValueError("Given fuera de rango.")

        previous = seen_givens.get((r, c))
        if previous is not None and previous != value:
            raise ValueError("Existen givens contradictorios.")

        seen_givens[(r, c)] = value

    for relation in data.get("inequalities", []):
        if not isinstance(relation, dict):
            raise ValueError("Cada desigualdad debe ser un diccionario.")

        cell1 = relation.get("cell1")
        cell2 = relation.get("cell2")
        operator = relation.get("operator")

        if operator not in ("<", ">"):
            raise ValueError("Operador de desigualdad inválido.")

        if (
            not isinstance(cell1, list)
            or not isinstance(cell2, list)
            or len(cell1) != 2
            or len(cell2) != 2
        ):
            raise ValueError("cell1 y cell2 deben tener formato [row, col].")

        r1, c1 = cell1
        r2, c2 = cell2

        if not all(type(x) is int for x in (r1, c1, r2, c2)):
            raise ValueError("Las coordenadas deben ser enteras.")

        if not (
            0 <= r1 < n
            and 0 <= c1 < n
            and 0 <= r2 < n
            and 0 <= c2 < n
        ):
            raise ValueError("Desigualdad fuera de rango.")

        if abs(r1 - r2) + abs(c1 - c2) != 1:
            raise ValueError("Las desigualdades solo pueden unir celdas adyacentes.")

    return True


def canonical_givens(data):
    return {
        (item["row"], item["col"], item["value"])
        for item in data.get("givens", [])
    }


def invert_operator(operator):
    return ">" if operator == "<" else "<"


def canonical_inequalities(data):
    result = set()

    for item in data.get("inequalities", []):
        r1, c1 = item["cell1"]
        r2, c2 = item["cell2"]
        operator = item["operator"]

        if r1 == r2 and c2 < c1:
            c1, c2 = c2, c1
            operator = invert_operator(operator)
        elif c1 == c2 and r2 < r1:
            r1, r2 = r2, r1
            operator = invert_operator(operator)

        result.add((r1, c1, operator, r2, c2))

    return result


def compare_instances(expected, predicted):
    expected_givens = canonical_givens(expected)
    predicted_givens = canonical_givens(predicted)
    expected_inequalities = canonical_inequalities(expected)
    predicted_inequalities = canonical_inequalities(predicted)

    return {
        "givens_exact": expected_givens == predicted_givens,
        "inequalities_exact": expected_inequalities == predicted_inequalities,
        "instance_exact": (
            expected_givens == predicted_givens
            and expected_inequalities == predicted_inequalities
        ),
        "missing_givens": len(expected_givens - predicted_givens),
        "extra_givens": len(predicted_givens - expected_givens),
        "missing_inequalities": len(
            expected_inequalities - predicted_inequalities
        ),
        "extra_inequalities": len(
            predicted_inequalities - expected_inequalities
        ),
    }
