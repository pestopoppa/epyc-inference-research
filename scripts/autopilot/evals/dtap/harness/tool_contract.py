"""Versioned native-tool schema contracts for opt-in DTAP endpoint runs.

The public schema file is deliberately separate from the imported upstream
cases, fixture arms, and deterministic judges.  It contains only published
MCP callable names, descriptions, and JSON Schema input contracts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .outcomes import HarnessFailure, ToolFailure

CONTRACT_VERSION = "dtap-native-tools-v1"
LEGACY_REQUEST_VERSION = "unadvertised-messages-only-v1"
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("tool_schemas.json")
MAX_TOOLS_PER_CASE = 128
MAX_SCHEMA_BYTES_PER_CASE = 256 * 1024
MAX_SCHEMA_DEPTH = 6
MAX_TOOL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 4096
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_TOOL_CALLS_PER_RESPONSE = 128
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SCHEMA_KEYS = {
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "anyOf", "description",
}
_SUPPORTED_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


@dataclass(frozen=True)
class NativeToolContract:
    version: str
    source_json: bytes
    cases_json: bytes
    sha256: str

    def __post_init__(self) -> None:
        # Enforce byte-backed storage even when constructed outside the loader.
        if not isinstance(self.source_json, bytes):
            object.__setattr__(self, "source_json", _canonical(dict(self.source_json)))
        if not isinstance(self.cases_json, bytes):
            object.__setattr__(self, "cases_json", _canonical(dict(self.cases_json)))

    @classmethod
    def from_mappings(
        cls, version: str, source: Mapping[str, str],
        cases: Mapping[str, List[Dict[str, Any]]], digest: str,
    ) -> "NativeToolContract":
        return cls(version, _canonical(dict(source)), _canonical(dict(cases)), digest)

    @property
    def source(self) -> Dict[str, str]:
        return json.loads(self.source_json)

    @property
    def cases(self) -> Dict[str, List[Dict[str, Any]]]:
        # Return fresh data so callers cannot mutate the loaded contract.
        return json.loads(self.cases_json)

    def for_case(self, case_id: str) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.cases_json)[case_id]
        except KeyError as exc:
            raise HarnessFailure(f"native tool contract has no case {case_id!r}") from exc

    def tools_sha256(self, case_id: str) -> str:
        return hashlib.sha256(_canonical(self.for_case(case_id))).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _check_schema(schema: Any, *, depth: int = 0, path: str = "parameters") -> None:
    if depth > MAX_SCHEMA_DEPTH:
        raise HarnessFailure(f"native tool schema exceeds maximum depth at {path}")
    if not isinstance(schema, dict):
        raise HarnessFailure(f"native tool schema must be an object at {path}")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown or "$ref" in schema:
        raise HarnessFailure(f"unsupported JSON Schema keys at {path}: {sorted(unknown | ({'$ref'} & set(schema)))}")
    if "type" in schema and schema["type"] not in _SUPPORTED_TYPES:
        raise HarnessFailure(f"unsupported JSON Schema type at {path}")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise HarnessFailure(f"enum must be an array at {path}")
    if "anyOf" in schema:
        if not isinstance(schema["anyOf"], list) or not schema["anyOf"]:
            raise HarnessFailure(f"anyOf must be a non-empty array at {path}")
        for index, child in enumerate(schema["anyOf"]):
            _check_schema(child, depth=depth + 1, path=f"{path}.anyOf[{index}]")
    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict) or any(not isinstance(k, str) for k in props):
            raise HarnessFailure(f"properties must map strings to schemas at {path}")
        for key, child in props.items():
            _check_schema(child, depth=depth + 1, path=f"{path}.{key}")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or any(not isinstance(k, str) for k in required):
            raise HarnessFailure(f"required must be an array of strings at {path}")
        if not set(required).issubset(set(schema.get("properties", {}))):
            raise HarnessFailure(f"required names must be declared properties at {path}")
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _check_schema(additional, depth=depth + 1, path=f"{path}.*")
    elif additional is not None and not isinstance(additional, bool):
        raise HarnessFailure(f"additionalProperties must be boolean or schema at {path}")
    if "items" in schema:
        _check_schema(schema["items"], depth=depth + 1, path=f"{path}[]")


def _validate_case_tools(case_id: str, tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list) or len(tools) > MAX_TOOLS_PER_CASE:
        raise HarnessFailure(f"tool catalog for {case_id!r} must be a bounded array")
    try:
        size = len(_canonical(tools))
    except (TypeError, ValueError) as exc:
        raise HarnessFailure(f"tool catalog for {case_id!r} is not canonical JSON") from exc
    if size > MAX_SCHEMA_BYTES_PER_CASE:
        raise HarnessFailure(f"tool catalog for {case_id!r} exceeds serialized size limit")
    names = set()
    for index, entry in enumerate(tools):
        if not isinstance(entry, dict) or set(entry) != {"type", "function"} or entry.get("type") != "function":
            raise HarnessFailure(f"tool catalog entry {index} for {case_id!r} is not a function schema")
        fn = entry.get("function")
        if not isinstance(fn, dict) or set(fn) != {"name", "description", "parameters"}:
            raise HarnessFailure(f"function schema entry {index} for {case_id!r} has invalid fields")
        name, description, params = fn["name"], fn["description"], fn["parameters"]
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or len(name) > MAX_TOOL_NAME_LENGTH:
            raise HarnessFailure(f"function schema entry {index} has an invalid plain function name")
        if name in names:
            raise HarnessFailure(f"duplicate native tool name {name!r} for {case_id!r}")
        names.add(name)
        if not isinstance(description, str) or len(description) > MAX_DESCRIPTION_LENGTH:
            raise HarnessFailure(f"function schema description for {name!r} is invalid or too long")
        _check_schema(params)
        if params.get("type") != "object" or params.get("additionalProperties") is not False:
            raise HarnessFailure(f"top-level arguments for {name!r} must be a closed object")
    return tools


def load_native_tool_contract(
    expected_cases: Any, path: Optional[Path] = None, *, registry_path: Optional[Path] = None,
) -> NativeToolContract:
    path = Path(path or DEFAULT_CONTRACT_PATH)
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessFailure(f"cannot load native tool contract: {exc}") from exc
    expected_case_ids = sorted(expected_cases) if isinstance(expected_cases, Mapping) else sorted(expected_cases)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "source", "case_registry", "server_catalogs", "case_servers", "cases",
    }:
        raise HarnessFailure("native tool contract has an invalid top-level structure")
    version = raw["schema_version"]
    if version != CONTRACT_VERSION:
        raise HarnessFailure(f"unsupported native tool contract version: {version!r}")
    source = raw["source"]
    if not isinstance(source, dict) or set(source) != {"repository", "commit", "tree"}:
        raise HarnessFailure("native tool contract source pin is missing or malformed")
    if any(not isinstance(source[key], str) or not source[key] for key in source):
        raise HarnessFailure("native tool contract source pin fields must be non-empty strings")
    case_registry = raw["case_registry"]
    if not isinstance(case_registry, dict) or set(case_registry) != {
        "repository", "commit", "path", "git_blob", "sha256",
    }:
        raise HarnessFailure("native tool contract case-registry pin is missing or malformed")
    if any(not isinstance(case_registry[key], str) or not case_registry[key] for key in case_registry):
        raise HarnessFailure("native tool contract case-registry pin fields must be non-empty strings")
    if registry_path is None:
        raise HarnessFailure("native tool contract requires the loaded case-registry file path")
    try:
        registry_bytes = Path(registry_path).read_bytes()
        registry_digest = hashlib.sha256(registry_bytes).hexdigest()
        registry_payload = json.loads(registry_bytes)
    except OSError as exc:
        raise HarnessFailure(f"cannot verify native tool case registry: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HarnessFailure(f"pinned native tool case registry is not valid JSON: {exc}") from exc
    if registry_digest != case_registry["sha256"]:
        raise HarnessFailure("native tool contract case-registry bytes do not match the pinned SHA-256")
    if not isinstance(registry_payload, dict) or not isinstance(registry_payload.get("cases"), dict):
        raise HarnessFailure("pinned case-registry bytes have an invalid cases mapping")
    if isinstance(expected_cases, Mapping) and registry_payload.get("cases") != dict(expected_cases):
        raise HarnessFailure("loaded case mapping differs from the pinned case-registry bytes")
    server_catalogs = raw["server_catalogs"]
    if not isinstance(server_catalogs, dict):
        raise HarnessFailure("native tool contract server catalogs are malformed")
    for server, names in server_catalogs.items():
        if not isinstance(server, str) or not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise HarnessFailure("native tool contract server catalog names are malformed")
        if len(names) != len(set(names)):
            raise HarnessFailure(f"duplicate function in server catalog {server!r}")
        if any(not _NAME_RE.fullmatch(name) for name in names):
            raise HarnessFailure(f"server catalog {server!r} contains a non-portable function name")
    case_servers = raw["case_servers"]
    if not isinstance(case_servers, dict) or set(case_servers) != set(expected_case_ids):
        raise HarnessFailure("native tool contract server mapping does not cover the expected cases")
    for case_id, entries in case_servers.items():
        if not isinstance(entries, list):
            raise HarnessFailure(f"native tool contract server mapping is malformed for {case_id!r}")
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != {"name", "tool_blacklist"}
                    or not isinstance(entry["name"], str)
                    or not isinstance(entry["tool_blacklist"], list)
                    or any(not isinstance(name, str) for name in entry["tool_blacklist"])):
                raise HarnessFailure(f"native tool contract server mapping is malformed for {case_id!r}")
    cases = raw["cases"]
    if not isinstance(cases, dict) or set(cases) != set(expected_case_ids):
        missing = sorted(set(expected_case_ids) - set(cases or {})) if isinstance(cases, dict) else sorted(expected_case_ids)
        extra = sorted(set(cases or {}) - set(expected_case_ids)) if isinstance(cases, dict) else []
        raise HarnessFailure(f"native tool contract case coverage mismatch (missing={missing}, extra={extra})")
    validated = {case_id: _validate_case_tools(case_id, cases[case_id]) for case_id in expected_case_ids}
    if isinstance(expected_cases, Mapping):
        for case_id, case in expected_cases.items():
            agent = (case.get("config") or {}).get("Agent") or {}
            expected_servers = [
                {"name": entry.get("name"), "tool_blacklist": sorted(entry.get("tool_blacklist", []))}
                for entry in agent.get("mcp_servers", []) if entry.get("enabled")
            ]
            if case_servers[case_id] != expected_servers:
                raise HarnessFailure(f"native tool contract server configuration is stale for {case_id!r}")
            expected_names = []
            for entry in expected_servers:
                server = entry["name"]
                if server not in server_catalogs:
                    raise HarnessFailure(f"no native function catalog for enabled server {server!r}")
                blocked = set(entry["tool_blacklist"])
                expected_names.extend(name for name in server_catalogs[server] if name not in blocked)
            actual_names = [tool["function"]["name"] for tool in validated[case_id]]
            if actual_names != expected_names:
                raise HarnessFailure(f"native tool schema/name mapping is stale for {case_id!r}")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return NativeToolContract.from_mappings(version, source, validated, digest)


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        return isinstance(value, float) and math.isfinite(value)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        raise ToolFailure(f"argument {path} must have type {expected}")
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        raise ToolFailure(f"argument {path} is not an allowed enum value")
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate_value(value, branch, path=path)
                break
            except ToolFailure:
                continue
        else:
            raise ToolFailure(f"argument {path} matches no allowed schema branch")
    if expected == "object":
        for key in schema.get("required", []):
            if key not in value:
                raise ToolFailure(f"missing required argument {path}.{key}")
        props = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, child_value in value.items():
            if key in props:
                _validate_value(child_value, props[key], path=f"{path}.{key}")
            elif additional is False:
                raise ToolFailure(f"unknown argument {path}.{key}")
            elif isinstance(additional, dict):
                _validate_value(child_value, additional, path=f"{path}.{key}")
    elif expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], path=f"{path}[{index}]")


def validate_tool_calls(calls: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> None:
    """Fail before accepted trace events when a call is undeclared or malformed."""
    if not isinstance(calls, list) or len(calls) > MAX_TOOL_CALLS_PER_RESPONSE:
        raise ToolFailure("tool-call response is not a bounded array")
    by_name = {entry["function"]["name"]: entry["function"]["parameters"] for entry in tools}
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ToolFailure(f"tool call {index} is not an object")
        name = call.get("name")
        if not isinstance(name, str) or name not in by_name:
            raise ToolFailure(f"tool call {index} used an undeclared function name")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            raise ToolFailure(f"arguments for tool {name!r} must be an object")
        try:
            encoded = _canonical(arguments)
        except (TypeError, ValueError) as exc:
            raise ToolFailure(f"arguments for tool {name!r} are not JSON-compatible") from exc
        if len(encoded) > MAX_ARGUMENT_BYTES:
            raise ToolFailure(f"arguments for tool {name!r} exceed the size limit")
        _validate_value(arguments, by_name[name], path=name)
