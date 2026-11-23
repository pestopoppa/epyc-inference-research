"""Loaded producer build identity and stable code projection, split verbatim from campaign_control.py (code-budget split)."""
from __future__ import annotations
import hashlib
import json
import types
from typing import Any
from . import runtime_aggregates as aggregates


def _stable_code_projection(code: types.CodeType) -> dict[str, Any]:
    """Return behavior-bearing code attributes without adaptive runtime state."""
    def constant(value: Any) -> Any:
        if isinstance(value, types.CodeType):
            return {"code": _stable_code_projection(value)}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, tuple):
            return {"tuple": [constant(item) for item in value]}
        if isinstance(value, frozenset):
            items = [constant(item) for item in value]
            return {"frozenset": sorted(
                items, key=lambda item: json.dumps(item, sort_keys=True))}
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return {"float": repr(value)}
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                "repr": repr(value)}

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "exceptiontable": code.co_exceptiontable.hex(),
        "consts": [constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _loaded_producer_build_identity(namespace: dict, module_name: str) -> dict[str, Any]:
    """Hash loaded callable bytecode and selected constants, with exact scope.

    `namespace`/`module_name` name the module whose code is identified (campaign_control, via its thin wrapper);
    the verbatim move into this file had silently narrowed the scope to THIS module's globals.
    """
    digest = hashlib.sha256()
    included: list[str] = []

    def add_callable(symbol: str, function: types.FunctionType, kind: str) -> None:
        if function.__code__.co_filename == "<string>":
            kind = f"generated_{kind}"
        label = f"{kind}:{symbol}"
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(json.dumps(
            _stable_code_projection(function.__code__), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode())
        included.append(label)

    for name, value in sorted(dict(namespace).items()):
        if isinstance(value, types.FunctionType) and value.__module__ == module_name:
            add_callable(name, value, "module_function")
        elif isinstance(value, type) and value.__module__ == module_name:
            for member_name, member in sorted(vars(value).items()):
                symbol = f"{name}.{member_name}"
                if isinstance(member, types.FunctionType):
                    add_callable(symbol, member, "method")
                elif isinstance(member, staticmethod):
                    add_callable(symbol, member.__func__, "staticmethod")
                elif isinstance(member, classmethod):
                    add_callable(symbol, member.__func__, "classmethod")
                elif isinstance(member, property):
                    for accessor, function in (("fget", member.fget),
                                               ("fset", member.fset),
                                               ("fdel", member.fdel)):
                        if function is not None:
                            add_callable(f"{symbol}.{accessor}", function,
                                         f"property_{accessor}")
        elif name.isupper() and isinstance(value, (str, int, float, frozenset)):
            label = f"constant:{name}"
            digest.update(label.encode())
            digest.update(b"\0")
            encoded = repr(sorted(value)) if isinstance(value, frozenset) else repr(value)
            digest.update(encoded.encode())
            included.append(label)
    # The installed aggregate validator is an explicit dependency, not captured
    # cache state. Its own identity covers all helper code and closed constants.
    digest.update(json.dumps(aggregates.source_identity(), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False).encode())
    included.append("installed_dependency:runtime_aggregates")
    return {"schema": "epyc.autokernel.loaded_producer_build.v1",
            "scope": "campaign_control_callable_bytecode_and_selected_constants",
            "module": module_name,
            "identity_basis": "loaded_callable_bytecode_and_constants_sha256",
            "included_symbols": included,
            "excluded_scope": ["campaign_service_transport", "journal_validator",
                               "whole_package_launch_attestation"],
            "sha256": digest.hexdigest()}
