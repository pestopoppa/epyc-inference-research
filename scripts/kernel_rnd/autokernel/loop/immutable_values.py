"""Shared conversions for immutable nested values."""
import copy
from types import MappingProxyType
from typing import Any, Mapping


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def freeze_copied(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({copy.deepcopy(key): freeze_copied(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_copied(item) for item in value)
    return copy.deepcopy(value)


def thaw_copied(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): thaw_copied(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_copied(item) for item in value]
    return copy.deepcopy(value)
