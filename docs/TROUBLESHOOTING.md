# Troubleshooting

## `sha256sum: ... No such file or directory`

The checksum file must be verified from the repository root:

```bash
sha256sum -c models/checksums.sha256
```

The entries in `models/checksums.sha256` include the `models/` prefix.

## `ModuleNotFoundError: No module named 'futoshiki_assistant'` in Colab

The final Colab notebook uses:

```python
%pip install -q .
```

instead of an editable install.

It also adds the repository `src/` directory to the current kernel path as a defensive fallback.

Verify with:

```python
import futoshiki_assistant
print("Package import OK")
```

## Protobuf warning in Colab

A warning involving preinstalled Google packages may appear when OR-Tools updates `protobuf`.

It is not considered a project failure if:

- package import succeeds;
- model checksum verification succeeds;
- `pytest -q` succeeds.
