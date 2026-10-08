"""Synthetic source conformance only; no models, processes, or runtime measurements."""
import copy
import itertools
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from . import graph_profile_capture as c

class CaptureTests(unittest.TestCase):
    def setUp(self):
        times = itertools.chain(("2026-01-01T00:00:00Z", "2026-01-01T00:00:02Z"), itertools.repeat("2026-01-01T00:00:04Z"))
        clock = patch.object(c, "writer_utc", side_effect=lambda: next(times))
        clock.start(); self.addCleanup(clock.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = {role: str(self.root / role) for role in c.ROLES}
        for role, path in self.raw.items():
            Path(path).write_text("\n".join(c.KNOBS) if role == "compiled_strings" else role)
        self.binary = self.root / "libggml-cpu.so"
        self.binary.write_bytes("\n".join(c.KNOBS).encode())
        h = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.p = {"capture_id": "synthetic-1", "owner": "synthetic-owner", "recorded_at": "2026-01-01T00:00:00Z",
                  "source": {"commit": "a"*40, "tree": "b"*40, "profiler_source_sha256": "c"*64},
                  "binary": {"path": str(self.binary), "sha256": h(self.binary), "mtime_ns": self.binary.stat().st_mtime_ns,
                             "build_commit": "a"*40, "compiled_knobs": list(c.KNOBS), "compiled_strings_sha256": h(self.raw["compiled_strings"])},
                  "knobs": dict.fromkeys(c.KNOBS), "threads": 2,
                  "graph": {"identity_sha256": h(self.raw["graph_identity"]), "identity_method": "owner_structural_receipt", "filter_semantics": "native_nnodes_eq_min_max_after_global_skip"},
                  "eval": {"skip": 0, "warmup_semantics": "skip_first_global_graph_evaluations_not_phase_detection", "accumulation_semantics": "post_skip_shape_filtered_node_index_aggregate", "measuring_thread": "thread0_wall_compute", "dispersion_semantics": "wall_max_accumulated_eval_and_threshold_spikes_no_thread_vector"},
                  "iqk": "disabled", "raw_paths": self.raw}
        self.raw["node_table"] = None
        self.p["outputs"] = {"node_table": {"availability": "unavailable", "path": None},
                             "log": {"availability": "retained_stderr", "path": self.raw["log"]},
                             "mm": {"availability": "unavailable", "path": None}}
        self.p["knobs"].update(GGML_CPU_PROF="1", GGML_CPU_PROF_PERNODE_FILE=self.raw["pernode"])
        self.p['measurement_metadata'] = {'schema':'epyc.graph_profile_measurement_metadata.v1',
            'date':'2026-01-01','category':'BASELINE','protocol_id':None,
            'metrics':[{'measurement_id':'synthetic-node-wall','metric':'node.wall_us',
                        'selector':{'kind':'node','identity':0,'field':'wall_us'},'metric_direction':'lower_better',
                        'unit':'us','claim':'Recorded instrumented thread-zero wall cost',
                        'reps_basis':'native_node_evals','attestation_role':'pernode'}]}
        self.o = {"started_at": "2026-01-01T00:00:01Z", "observed_at": "2026-01-01T00:00:02Z", "owner": self.p["owner"], "capture_id": self.p["capture_id"], "binary_sha256": h(self.binary), "effective_knobs": self.p["knobs"], "effective_threads": 2, "graph_identity_sha256": self.p["graph"]["identity_sha256"], "thread_availability": "unavailable"}
        self.closed = {"ended_at": "2026-01-01T00:00:03Z", "closed_at": "2026-01-01T00:00:04Z", "producer_exited": True, "all_handles_closed": True, "owner": self.p["owner"], "capture_id": self.p["capture_id"]}
        self.pre, self.during, self.seal = (self.root / name for name in ("pre.json", "during.json", "seal.json"))

    def phases(self):
        c.begin(self.pre, self.p)
        c.during(self.during, self.pre, self.o)

    def test_roundtrip_and_exclusive_seal(self):
        self.phases()
        c.finalize(self.seal, self.pre, self.during, self.closed)
        envelope = json.loads(self.seal.read_text())
        self.assertEqual(envelope["observation"]["thread_availability"], "unavailable")
        self.assertEqual(envelope["raw"]["pernode"]["size"], len(b"pernode"))
        with self.assertRaises(FileExistsError):
            c.finalize(self.seal, self.pre, self.during, self.closed)

    def test_required_provenance_refusals(self):
        for key in self.p:
            bad = copy.deepcopy(self.p); del bad[key]
            with self.assertRaises(c.Refusal): c.begin(self.pre, bad)
        self.assertFalse(self.pre.exists())

    def test_unknown_and_inconsistent_pre(self):
        cases = [("iqk", "unknown"), ("threads", True), ("owner", ""), ("recorded_at", "unknown")]
        for key, value in cases:
            bad = copy.deepcopy(self.p); bad[key] = value
            with self.assertRaises(c.Refusal): c.begin(self.pre, bad)
        bad = copy.deepcopy(self.p); bad["graph"]["identity_method"] = "node_count"
        with self.assertRaises(c.Refusal): c.begin(self.pre, bad)

    def test_during_drift_thread_unavailable_and_window(self):
        c.begin(self.pre, self.p)
        for key, value in (("effective_threads", 3), ("thread_availability", "aggregate_max_mean_min"), ("started_at", "2025-01-01T00:00:00Z"), ("owner", "other")):
            bad = copy.deepcopy(self.o); bad[key] = value
            with self.assertRaises(c.Refusal): c.during(self.during, self.pre, bad)
        self.assertFalse(self.during.exists())

    def test_closure_and_mutation_refusals(self):
        self.phases()
        for key in ("producer_exited", "all_handles_closed"):
            bad = dict(self.closed); bad[key] = False
            with self.assertRaises(c.Refusal): c.finalize(self.seal, self.pre, self.during, bad)
        self.binary.write_bytes(b"changed")
        with self.assertRaises(c.Refusal): c.finalize(self.seal, self.pre, self.during, self.closed)
        self.assertFalse(self.seal.exists())

    def test_raw_symlink_refusal(self):
        self.phases()
        path = Path(self.raw["log"]); path.unlink(); path.symlink_to(self.binary)
        with self.assertRaises(c.Refusal): c.finalize(self.seal, self.pre, self.during, self.closed)

    def test_missing_compiled_knob_refusal(self):
        self.phases()
        Path(self.raw["compiled_strings"]).write_text("GGML_CPU_PROF")
        with self.assertRaises(c.Refusal): c.finalize(self.seal, self.pre, self.during, self.closed)

    def test_duplicate_pre_field_refusal(self):
        self.pre.write_text('{"schema":"a","schema":"b"}')
        with self.assertRaises(c.Refusal): c.during(self.during, self.pre, self.o)

    def test_known_semantics_refuse_unknown_and_arbitrary_text(self):
        for section, fields in (("graph", ("filter_semantics",)), ("eval", ("warmup_semantics", "accumulation_semantics", "dispersion_semantics"))):
            for field in fields:
                for value in ("unknown", "plausible but unspecified", ""):
                    bad = copy.deepcopy(self.p); bad[section][field] = value
                    with self.assertRaises(c.Refusal): c.begin(self.pre, bad)
        self.assertFalse(self.pre.exists())

    def test_pre_byte_identity_refusals(self):
        for role in ("binary", "graph_identity", "compiled_strings"):
            path = self.binary if role == "binary" else Path(self.raw[role])
            original = path.read_bytes(); path.write_bytes(original + b"changed")
            with self.assertRaises(c.Refusal): c.begin(self.pre, self.p)
            path.write_bytes(original)
            if role == "binary": self.p["binary"]["mtime_ns"] = path.stat().st_mtime_ns
        self.assertFalse(self.pre.exists())

    def test_pre_missing_compiled_string_even_matching_digest(self):
        self.binary.write_bytes(b"GGML_CPU_PROF")
        self.p["binary"].update(sha256=hashlib.sha256(self.binary.read_bytes()).hexdigest(), mtime_ns=self.binary.stat().st_mtime_ns)
        with self.assertRaises(c.Refusal): c.begin(self.pre, self.p)

    def test_gated_outputs_and_presence_zero_threads(self):
        self.p["knobs"].update(GGML_CPU_PROF_NODES="0", GGML_CPU_PROF_MM="0", GGML_CPU_PROF_THREADS="0")
        self.raw["node_table"] = self.raw["log"]
        self.p["outputs"]["node_table"] = {"availability": "requested_stderr", "path": self.raw["log"]}
        self.p["outputs"]["mm"] = {"availability": "requested_stderr", "path": self.raw["log"]}
        self.o["thread_availability"] = "aggregate_max_mean_min"
        self.phases(); c.finalize(self.seal, self.pre, self.during, self.closed)
        self.assertEqual(json.loads(self.seal.read_text())["raw"]["node_table"], json.loads(self.seal.read_text())["raw"]["log"])

    def test_absent_table_and_mm_cannot_claim_available(self):
        for output in ("node_table", "mm"):
            bad = copy.deepcopy(self.p); bad["outputs"][output] = {"availability": "requested_stderr", "path": self.raw["log"]}
            with self.assertRaises(c.Refusal): c.begin(self.pre, bad)

    def test_separate_table_file_contract(self):
        table = str(self.root / "separate-table")
        Path(table).write_text("synthetic table")
        self.p["knobs"].update(GGML_CPU_PROF_NODES="1", GGML_CPU_PROF_NODES_FILE=table)
        self.raw["node_table"] = table
        self.p["outputs"]["node_table"] = {"availability": "requested_file", "path": table}
        self.phases(); c.finalize(self.seal, self.pre, self.during, self.closed)
        self.assertEqual(json.loads(self.seal.read_text())["raw"]["node_table"]["path"], table)

    def test_native_numeric_gate_refusals(self):
        for key in ("GGML_CPU_PROF_SKIP", "GGML_CPU_PROF_NNODES_EQ", "GGML_CPU_PROF_NNODES_MIN", "GGML_CPU_PROF_NNODES_MAX", "GGML_CPU_PROF_SPIKE_US"):
            for value in ("unknown", "-1", "1junk", "01", "2147483648"):
                bad = copy.deepcopy(self.p); bad["knobs"][key] = value
                with self.assertRaises(c.Refusal): c.begin(self.pre, bad)
        bad = copy.deepcopy(self.p); bad["knobs"].update(GGML_CPU_PROF_NNODES_MIN="3", GGML_CPU_PROF_NNODES_MAX="2")
        with self.assertRaises(c.Refusal): c.begin(self.pre, bad)

    def test_path_device_and_inode_both_bound(self):
        actual_stat = c.os.stat
        binary = str(self.binary)
        def changed_device(path, *args, **kwargs):
            st = actual_stat(path, *args, **kwargs)
            if str(path) == binary and kwargs.get("follow_symlinks") is False:
                return SimpleNamespace(st_dev=st.st_dev+1, st_ino=st.st_ino, st_mode=st.st_mode)
            return st
        with patch.object(c.os, "stat", side_effect=changed_device):
            with self.assertRaises(c.Refusal): c.receipt(self.binary)

    def test_finalize_has_no_temporary_phase_write(self):
        self.phases()
        with patch.object(c, "during", side_effect=AssertionError("finalize must validate without writing")):
            c.finalize(self.seal, self.pre, self.during, self.closed)

    def test_historical_window_cannot_be_backfilled_now(self):
        with patch.object(c, "writer_utc", return_value="2026-10-08T00:00:00Z"):
            c.begin(self.pre, self.p)
            with self.assertRaises(c.Refusal): c.during(self.during, self.pre, self.o)
        self.assertFalse(self.during.exists())

    def test_actual_writer_timestamps_and_future_window_refusal(self):
        self.phases()
        with patch.object(c, "writer_utc", return_value="2026-01-01T00:00:03Z"):
            with self.assertRaises(c.Refusal): c.finalize(self.seal, self.pre, self.during, self.closed)
        self.assertFalse(self.seal.exists())
        with patch.object(c, "writer_utc", return_value="2026-01-01T00:00:04Z"):
            c.finalize(self.seal, self.pre, self.during, self.closed)
        self.assertEqual(json.loads(self.pre.read_text())["writer_observed_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(json.loads(self.during.read_text())["writer_observed_at"], "2026-01-01T00:00:02Z")
        self.assertEqual(json.loads(self.seal.read_text())["writer_observed_at"], "2026-01-01T00:00:04Z")

    def test_during_owner_observation_cannot_be_in_writer_future(self):
        c.begin(self.pre, self.p)
        with patch.object(c, "writer_utc", return_value="2026-01-01T00:00:01Z"):
            with self.assertRaises(c.Refusal): c.during(self.during, self.pre, self.o)
        self.assertFalse(self.during.exists())

    def test_measurement_metadata_required_no_unknown_labels(self):
        for key in self.p['measurement_metadata']:
            bad=copy.deepcopy(self.p);del bad['measurement_metadata'][key]
            with self.assertRaises(c.Refusal):c.begin(self.pre,bad)
        for key,value in (('category','unknown'),('date','unknown'),('protocol_id','')):
            bad=copy.deepcopy(self.p);bad['measurement_metadata'][key]=value
            with self.assertRaises(c.Refusal):c.begin(self.pre,bad)

    def test_native_metric_identity_direction_units_and_basis_refuse(self):
        for key,value in (('measurement_id',''),('metric_direction','unknown'),('unit','seconds'),('reps_basis','independent_experiments'),('attestation_role','invented')):
            bad=copy.deepcopy(self.p);bad['measurement_metadata']['metrics'][0][key]=value
            with self.assertRaises(c.Refusal):c.begin(self.pre,bad)

    def test_missing_gradable_metadata_stays_explicit_absent(self):
        self.p['measurement_metadata'].update(date=None,protocol_id=None)
        self.p['measurement_metadata']['metrics'][0]['attestation_role']=None
        self.phases();c.finalize(self.seal,self.pre,self.during,self.closed)
        metadata=json.loads(self.seal.read_text())['provenance']['measurement_metadata']
        self.assertIsNone(metadata['date']);self.assertIsNone(metadata['protocol_id'])
        self.assertIsNone(metadata['metrics'][0]['attestation_role'])

    def test_recorded_measurement_date_must_match_capture_start(self):
        self.p['measurement_metadata']['date']='2030-01-01'
        c.begin(self.pre,self.p)
        with self.assertRaises(c.Refusal):c.during(self.during,self.pre,self.o)
        self.assertFalse(self.during.exists())
