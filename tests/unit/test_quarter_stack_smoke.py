"""Focused tests for the deterministic both-mode production smoke."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import yaml

from scripts.server.stack_manifest import HOT_SERVERS, PORT_MAP, ROLE_LAUNCH_META
from scripts.server.stack_numa import NUMA_CONFIG
from scripts.smoke import quarter_stack_smoke as smoke

# Ports the stack used to expose and no longer does. 8280/8380 and 8282/8382 were
# frontdoor and worker_general quarters 4-5; 8385/8485 were ingest_long_context
# quarters 4-5; 8087 was the standalone vision_escalation 7B, now an alias on
# worker_vision's :8086 process. The hand-maintained EXPECTED_CHAT_PORTS literal
# still probed all seven until 2026-08-01.
RETIRED_CHAT_PORTS = frozenset({8280, 8380, 8282, 8382, 8385, 8485, 8087})

REGISTRY = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"


def _registry_server_mode() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["server_mode"]


class FakeResponse:
    def __init__(self, body: object, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=None, response=None)

    def json(self) -> object:
        return self.body


def test_topology_sources_are_coherent() -> None:
    assert smoke.topology_errors() == []


def test_chat_ports_are_derived_from_the_launcher_hot_tier() -> None:
    """The chat surface IS the manifest's computed HOT tier minus the embedders."""
    assert smoke.EXPECTED_CHAT_PORTS == tuple(
        server["port"] for server in HOT_SERVERS if not server.get("embedding")
    )
    assert smoke.EXPECTED_CHAT_PORTS == smoke.expected_chat_ports()
    assert smoke.EXPECTED_CHAT_PORTS  # non-empty
    assert len(set(smoke.EXPECTED_CHAT_PORTS)) == len(smoke.EXPECTED_CHAT_PORTS)


def test_derived_chat_ports_drop_retired_ports_and_pick_up_live_ones() -> None:
    """Regression pin for the drift the hand-maintained literal had accumulated."""
    derived = set(smoke.EXPECTED_CHAT_PORTS)

    assert derived & RETIRED_CHAT_PORTS == set()
    # Every NUMA-pinned chat instance the stack actually declares is probed.
    # 2026-09-26: this named frontdoor / worker_general / ingest_long_context.
    # The operator-signed 2026-09-22 cutover (orchestrator 860b0b2d) made the
    # latter two ALIASES (of frontdoor's :8070 CPU fleet and architect_general's
    # :8083 GPU process) and deleted their NUMA_CONFIG entries, so the host set
    # is now DERIVED: every registry server_mode row that owns a process (no
    # alias_of) and has a NUMA_CONFIG entry. (eval_batch_frontdoor is a NUMA
    # role with no server_mode row — an eval-tower lane, not the HOT stack.)
    # STACKCHG-DFLASH2-20261003 (operator-signed): worker_vision (:8086) is now a
    # COLD CPU role (launch tier warm) that a default `start` never launches, so the
    # smoke must NOT probe it — probing it would report a deliberately-stopped
    # server as down. Hosts therefore split by launch tier: every HOT host's
    # instances are probed, and every non-HOT host's instances are not.
    server_mode = _registry_server_mode()
    hosts_by_tier: dict[str, list[str]] = {}
    for role, row in sorted(server_mode.items()):
        if isinstance(row, dict) and not row.get("alias_of") and role in NUMA_CONFIG:
            tier = ROLE_LAUNCH_META[role]["tier"]
            hosts_by_tier.setdefault(tier, []).append(role)
    hosts = hosts_by_tier.get("hot", [])
    assert "frontdoor" in hosts and len(hosts) >= 3, hosts  # non-vacuity
    probed = 0
    for role in hosts:
        for _cpus, port, _threads in NUMA_CONFIG[role]["instances"]:
            assert port in derived, f"{role} instance on {port} is not probed"
            probed += 1
    assert probed > len(hosts), "expected at least one multi-instance host fleet"
    cold_hosts = [role for tier, roles in hosts_by_tier.items() if tier != "hot"
                  for role in roles]
    assert "worker_vision" in cold_hosts, hosts_by_tier  # non-vacuity of the cold arm
    for role in cold_hosts:
        for _cpus, port, _threads in NUMA_CONFIG[role]["instances"]:
            assert port not in derived, f"cold {role} instance on {port} is probed"
    # Aliases add no port: each resolves to its host's probed port.
    aliases = {r: row["alias_of"] for r, row in server_mode.items() if row.get("alias_of")}
    assert "ingest_long_context" in aliases, aliases
    for alias, host in aliases.items():
        if alias in PORT_MAP:
            assert PORT_MAP[alias] == PORT_MAP[host] and PORT_MAP[alias] in derived, alias
    assert PORT_MAP["worker_general"] in derived
    # The :8074 full-CPU instance went HOT on 2026-08-01 (as architect_critic);
    # the literal never listed it. It serves architect_general since the
    # 2026-09-27 ARCHITECT SWAP.
    assert PORT_MAP["architect_general"] in derived
    assert PORT_MAP["architect_critic"] in derived
    # Aliases share their host process's port rather than adding one.
    assert PORT_MAP["vision_escalation"] == PORT_MAP["worker_vision"]
    assert PORT_MAP["coder_escalation"] == PORT_MAP["architect_critic"]


def test_topology_errors_flags_a_role_that_port_map_disagrees_about(monkeypatch) -> None:
    """The coherence check has teeth: PORT_MAP drift must be reported."""
    monkeypatch.setitem(smoke.PORT_MAP, "frontdoor", 9999)

    errors = smoke.topology_errors()

    assert any("frontdoor" in error and "9999" in error for error in errors)


def test_run_smoke_is_sequential_and_writes_one_row_per_endpoint(tmp_path, monkeypatch) -> None:
    seen: list[str] = []

    def post(url: str, **_kwargs: object) -> FakeResponse:
        seen.append(url)
        if url.endswith("/embedding"):
            return FakeResponse({"embedding": [0.0] * 1024})
        return FakeResponse(
            {"choices": [{"message": {"content": "ok", "reasoning_content": None}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr(smoke.httpx, "post", post)
    output = tmp_path / "nested" / "smoke.jsonl"

    assert smoke.run_smoke(output) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    expected_ports = list(smoke.EXPECTED_CHAT_PORTS + smoke.EXPECTED_EMBEDDER_PORTS)
    assert len(rows) == len(expected_ports)
    assert [row["port"] for row in rows] == expected_ports
    assert all(row["ok"] for row in rows)
    assert seen == [row["url"] for row in rows]


def test_embedding_row_accepts_llama_cpp_array_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke.httpx,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            [{"index": 0, "embedding": [[0.0] * smoke.EMBEDDING_DIMENSION]}]
        ),
    )

    row = smoke._embedding_row(8090, 1.0)

    assert row["ok"] is True
    assert row["dimension"] == smoke.EMBEDDING_DIMENSION


def test_endpoint_failure_is_recorded_without_fail_fast(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def post(url: str, **_kwargs: object) -> FakeResponse:
        calls.append(url)
        if ":8080/" in url:
            return FakeResponse({}, status_code=503)
        if url.endswith("/embedding"):
            return FakeResponse({"data": [{"embedding": [0.0] * 1024}]})
        return FakeResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(smoke.httpx, "post", post)
    output = tmp_path / "smoke.jsonl"

    assert smoke.run_smoke(output) == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == len(smoke.EXPECTED_CHAT_PORTS) + len(smoke.EXPECTED_EMBEDDER_PORTS)
    assert rows[0]["port"] == 8070 and rows[0]["ok"] is True
    assert rows[1]["port"] == 8080 and rows[1]["ok"] is False
    assert rows[-1]["port"] == 8095 and rows[-1]["ok"] is True


def test_topology_drift_fails_without_requests_and_publishes_empty_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(smoke, "EMBEDDER_PORTS", [8090])

    def post(*_args: object, **_kwargs: object) -> FakeResponse:
        raise AssertionError("must not request")

    monkeypatch.setattr(smoke.httpx, "post", post)
    output = tmp_path / "smoke.jsonl"

    assert smoke.run_smoke(output) == 2
    assert output.read_text(encoding="utf-8") == ""
