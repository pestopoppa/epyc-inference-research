"""AutoKernel lineage/fold manager (AK parallel-loop plan P4).

Deterministic keep ledger -> next C-composite fold list + presence-gate manifest +
staged Spec-D re-check rows. See `ledger.py` for the rules and `__main__.py` for the CLI.
"""
from .ledger import (FoldRefused, Feature, build_ledger, fold_entry, fold_list,
                     presence_manifest, specd_rows)

__all__ = ["FoldRefused", "Feature", "build_ledger", "fold_entry", "fold_list",
           "presence_manifest", "specd_rows"]
