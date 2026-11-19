"""AKX-P0d/P1d: census store, freshness-key derivation, T0/T1 witnesses, hook wiring."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ..evaluator import blast_radius_controls as C
from . import blast_radius as BR
from . import census as CE
from . import census_store as CS
from . import keep_screen as KS

CHAMP = "a" * 40


def census_doc(wid, sig, *, champion=CHAMP, probe=None, profile="x", types=("Q8_0", "F32"),
               observed=True):
    ident = {"workload_id": wid, "workload_signature": sig,
             **CS.census_identity(champion_commit=champion, probe=probe)}
    shape = {"phase": "prefill", "n_tokens": 8, "n_seq": 1}
    row = {"state": CE.OBSERVED if observed else CE.UNKNOWN, "nodes_total": CE.MIN_GRAPH_NODES + 5,
           "device_seen": False, "shape": shape, "op_backend": {"MUL_MAT": {"CPU": 3, profile: 1}}}
    return {"schema": CE.SCHEMA, "identity": ident,
            "weights": {"tensor_types": {t: 1 for t in types}},
            "graph": {"shape_envelope": [shape], "per_shape": [row]},
            "validity": {"freshness_key": CE.freshness_key(ident)}}


class Store(unittest.TestCase):
    def test_ensure_store_is_idempotent_and_marks_the_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = CS.ensure_store(Path(td) / "a" / "census")
            self.assertEqual(json.loads((root / "STORE.json").read_text())["schema"], CS.STORE_SCHEMA)
            (root / "STORE.json").write_text('{"kept": 1}')
            CS.ensure_store(root)
            self.assertEqual(json.loads((root / "STORE.json").read_text()), {"kept": 1})

    def test_env_overrides_the_default_store(self):
        with mock.patch.dict(os.environ, {CS.STORE_ENV: "/tmp/x-census"}):
            self.assertEqual(CS.store_path(), Path("/tmp/x-census"))
        with mock.patch.dict(os.environ, clear=True):
            self.assertEqual(CS.store_path(), CE.DEFAULT_CENSUS_STORE)

    def test_expected_key_equals_the_key_a_producer_stamps(self):
        targets = [{"target_ids": ["W3"], "workload_signature": "sigW3"}]
        doc = census_doc("W3", "sigW3")
        self.assertEqual(CS.expected_keys(targets, champion_commit=CHAMP)["W3"],
                         doc["validity"]["freshness_key"])
        self.assertNotEqual(CS.expected_keys(targets, champion_commit="b" * 40)["W3"],
                            doc["validity"]["freshness_key"])
        self.assertNotEqual(CS.expected_keys(targets, champion_commit=CHAMP, probe="p")["W3"],
                            doc["validity"]["freshness_key"])
        with self.assertRaises(ValueError):
            CS.census_identity(champion_commit="")

    def test_store_round_trip_only_fresh_records_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = CS.ensure_store(Path(td) / "c")
            CE.store_census(census_doc("W3", "s3"), root)
            CE.store_census(census_doc("W4", "s4", champion="b" * 40), root)  # stale champion
            targets = [{"target_ids": ["W3"], "workload_signature": "s3"},
                       {"target_ids": ["W4"], "workload_signature": "s4"}]
            got = CS.censuses_for_keep(targets, champion_commit=CHAMP, store=root)
            self.assertEqual(sorted(got), ["W3"])
            self.assertTrue(got["W3"][0].census_observed)
            self.assertEqual(CS.censuses_for_keep(targets, champion_commit=CHAMP,
                                                  store=Path(td) / "absent"), {})
            self.assertEqual(CS.censuses_for_keep(targets, champion_commit="", store=root), {})


class Witnesses(unittest.TestCase):
    def test_dso_digests_hash_real_libs_and_follow_symlinks_once(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "libggml-cpu.so.0.16.0").write_bytes(b"cpu")
            (d / "libggml-cpu.so.0").symlink_to("libggml-cpu.so.0.16.0")
            (d / "libggml-cpu.so").symlink_to("libggml-cpu.so.0")
            (d / "libother.so").write_bytes(b"x")
            (d / "llama-server").write_bytes(b"bin")
            got = CS.dso_digests(d)
            self.assertEqual(list(got), ["libggml-cpu.so"])
            self.assertEqual(got["libggml-cpu.so"], (
                __import__("hashlib").sha256(b"cpu").hexdigest()))
            self.assertEqual(CS.dso_digests(d / "none"), {})

    def test_t1_needs_both_observed_on_one_envelope(self):
        a, b = census_doc("W3", "s"), census_doc("W3", "s")
        w = CS.t1_witness(a, b)
        self.assertEqual(w[0], w[1])
        self.assertEqual(len(w[2]), 64)
        changed = census_doc("W3", "s", profile="IQ3")
        self.assertNotEqual(CS.t1_witness(a, changed)[0], CS.t1_witness(a, changed)[1])
        self.assertIsNone(CS.t1_witness(a, census_doc("W3", "s", observed=False)))
        other = census_doc("W3", "s")
        other["graph"]["shape_envelope"] = [{"phase": "decode", "n_tokens": 1, "n_seq": 1}]
        self.assertIsNone(CS.t1_witness(a, other))

    def test_with_witnesses_feeds_the_production_classifier(self):
        doc = census_doc("W2", "s2", types=("Q4_K", "F32"))
        rec, key = BR.workload_record_from_census(doc, expected_freshness_key=doc["validity"]["freshness_key"])
        self.assertEqual(rec.t1_profiles, None)
        w = CS.with_witnesses(rec, t1=CS.t1_witness(doc, doc))
        fp = C.produce(C.Q8_0_DISPATCH)
        row = BR.classify(fp, w, keep_id="k", census_freshness_key=key, created_at="t")
        self.assertEqual((row["classification"], row["tier"]), ("INERT", "T1"))
        bare = BR.classify(fp, rec, keep_id="k", census_freshness_key=key, created_at="t")
        self.assertEqual(bare["classification"], "UNKNOWN")
        same = {"libggml-cpu.so": "1" * 64}
        t0 = CS.with_witnesses(rec, t0=(same, dict(same)))
        row = BR.classify(C.produce(C.COMMENT_ONLY), t0, keep_id="k", census_freshness_key=key,
                          created_at="t")
        self.assertEqual((row["classification"], row["tier"]), ("INERT", "T0"))
        diff = CS.with_witnesses(rec, t0=(same, {"libggml-cpu.so": "2" * 64}))
        self.assertNotEqual(BR.classify(C.produce(C.COMMENT_ONLY), diff, keep_id="k",
                                        census_freshness_key=key, created_at="t")["classification"],
                            "INERT")


class HookWiring(unittest.TestCase):
    def _campaign(self, sig):
        targets = [{"target_ids": ["W3"], "workload_signature": sig, "status": "ready"}]
        return SimpleNamespace(targets=[SimpleNamespace(to_dict=lambda t=t: t) for t in targets],
                               campaign_id="c", manifest_digest=C.sha("manifest"))

    def test_hook_uses_a_fresh_store_census_and_ignores_a_stale_one(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b",
                                             "-c", "user.name=a", *a], check=True,
                                            capture_output=True, text=True).stdout.strip()
            git("init", "-q")
            (repo / "k.c").write_text("int f(void) { return 0; }\n")
            git("add", "k.c")
            git("commit", "-qm", "base")
            (repo / "k.c").write_text("// note\nint f(void) { return 0; }\n")
            git("commit", "-qam", "keep")
            manifest = root / "campaign.json"
            manifest.write_text("{}")
            store = CS.ensure_store(root / "census")
            CE.store_census(census_doc("W3", C.sha("sig-W3")), store)
            camp = self._campaign(C.sha("sig-W3"))

            def screen(champion):
                path = BR.record_keep_best_effort(
                    root / ("out-" + champion[:4]), repo, keep_id="k", child=git("rev-parse", "HEAD"),
                    resolved_campaign=camp, manifest_path=str(manifest),
                    champion_commit=champion, census_store=store)
                body = KS._read_record(path)
                return body["rows"][0]["census_freshness_key"]
            fresh = screen(CHAMP)
            self.assertFalse(fresh.startswith(BR.ABSENT_CENSUS))
            self.assertTrue(screen("c" * 40).startswith(BR.ABSENT_CENSUS))
            # no champion given: the historical behaviour (no census), never an error
            path = BR.record_keep_best_effort(root / "out-none", repo, keep_id="k",
                                              child=git("rev-parse", "HEAD"),
                                              resolved_campaign=camp, manifest_path=str(manifest))
            self.assertTrue(KS._read_record(path)["rows"][0]["census_freshness_key"]
                            .startswith(BR.ABSENT_CENSUS))
