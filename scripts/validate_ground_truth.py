import json
from pathlib import Path

from futoshiki_assistant.cp.solver import check_uniqueness
from futoshiki_assistant.schema import validate_instance


def main():
    root = Path(
        "data/evaluation/ground_truth"
    )

    failures = []

    for path in sorted(
        root.glob("*.json")
    ):
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        validate_instance(data)

        result = check_uniqueness(data)

        expected = data.get(
            "expected_status"
        )

        matches = (
            expected
            ==
            result["status"]
        )

        print(
            f"{path.name}: "
            f"expected={expected}, "
            f"actual={result['status']}, "
            f"match={matches}"
        )

        if not matches:
            failures.append(
                path.name
            )

    if failures:
        raise SystemExit(
            "Ground truth validation failed: "
            +
            ", ".join(
                failures
            )
        )


if __name__ == "__main__":
    main()
