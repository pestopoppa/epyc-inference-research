"""Stable ``python -m`` shim; implementation remains imported under its canonical name."""
from .source_build_worker import main


if __name__ == "__main__":
    raise SystemExit(main())

