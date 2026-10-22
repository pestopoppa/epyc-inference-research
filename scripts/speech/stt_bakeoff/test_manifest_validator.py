import copy
import json

import pytest

from scripts.speech.stt_bakeoff import manifest_validator as mv
from scripts.speech.stt_bakeoff.manifest_validator import ManifestRefused

H = "a" * 64
H2 = "b" * 64


def good(tmp_path, label="public-anchor"):
    m = {"set_id": "s1", "label": label, "track": "verbatim", "language": "en", "reference_version": 1,
         "utterances": [{"utterance_id": "u1", "pcm_sha256": H, "reference_text_sha256": H, "fillers_in_ref": False},
                        {"utterance_id": "u2", "pcm_sha256": H2, "reference_text_sha256": H2, "fillers_in_ref": True}]}
    if label == "private":
        a = tmp_path / "data" / "audio"
        r = tmp_path / "data" / "ref"
        a.mkdir(parents=True)
        r.mkdir(parents=True)
        m["audio_root"], m["reference_root"] = str(a), str(r)
    return m


def reason(m, prior=None):
    with pytest.raises(ManifestRefused) as e:
        mv.validate_manifest(m, prior)
    return e.value.reason


def test_good_public_and_private(tmp_path):
    mv.validate_manifest(good(tmp_path))
    mv.validate_manifest(good(tmp_path, "private"))


def test_unlabeled_refused(tmp_path):
    m = good(tmp_path)
    del m["label"]
    assert reason(m) == "missing_label"
    m["label"] = "semi"
    assert reason(m) == "bad_label"


def test_private_in_git_tree_refused(tmp_path):
    m = good(tmp_path, "private")
    (tmp_path / ".git").mkdir()
    assert reason(m) == "private_path_in_git_tree"


def test_private_in_git_file_worktree_refused(tmp_path):
    m = good(tmp_path, "private")
    (tmp_path / ".git").write_text("gitdir: /elsewhere")
    assert reason(m) == "private_path_in_git_tree"


def test_private_reference_path_in_git_refused(tmp_path):
    m = good(tmp_path, "private")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    m["reference_root"] = str(repo / "refs")
    assert reason(m) == "private_path_in_git_tree"


def test_reference_edit_without_bump(tmp_path):
    old = good(tmp_path)
    new = copy.deepcopy(old)
    new["utterances"][0]["reference_text_sha256"] = "c" * 64
    assert reason(new, old) == "reference_edit_without_bump"
    new["reference_version"] = 2
    assert reason(new, old) == "reference_edit_not_logged"
    new["edit_log"] = [{"reference_version": 2, "note": "pass 2"}]
    mv.validate_manifest(new, old)


def test_unchanged_refs_need_no_bump(tmp_path):
    old = good(tmp_path)
    mv.validate_manifest(copy.deepcopy(old), old)


def test_raw_content_refused(tmp_path):
    m = good(tmp_path)
    m["utterances"][0]["reference_text"] = "hello"
    assert reason(m) == "raw_content_in_manifest"
    m = good(tmp_path)
    m["utterances"][0]["audio_bytes"] = b"\x00"
    assert reason(m) == "raw_content_in_manifest"


@pytest.mark.parametrize("bad", ["A" * 64, "a" * 63, "g" * 64, "", None])
def test_bad_pcm_sha(tmp_path, bad):
    m = good(tmp_path)
    m["utterances"][0]["pcm_sha256"] = bad
    assert reason(m) == "bad_pcm_sha256"


def test_confirmation_needs_frozen_hash_and_rater(tmp_path):
    m = good(tmp_path, "private")
    m["confirmation"] = True
    assert reason(m) == "missing_rater_label"
    m["reference_protocol_label"] = "single-rater"
    m["rater_count"] = 1
    assert reason(m) == "confirmation_missing_frozen_hash"
    m["frozen_manifest_sha256"] = "d" * 64
    assert reason(m) == "confirmation_frozen_hash_mismatch"
    m["frozen_manifest_sha256"] = mv.canonical_manifest_sha256(m)
    mv.validate_manifest(m)
    m["utterances"][0]["reference_text_sha256"] = "e" * 64  # edit after freeze
    assert reason(m) == "confirmation_frozen_hash_mismatch"


def test_mixed_strata_and_tracks(tmp_path):
    pub, priv = good(tmp_path), good(tmp_path, "private")
    mv.check_no_mixed_strata([pub, copy.deepcopy(pub)])
    with pytest.raises(ManifestRefused) as e:
        mv.check_no_mixed_strata([pub, priv])
    assert e.value.reason == "mixed_strata"
    cl = copy.deepcopy(pub)
    cl["track"] = "cleaned"
    with pytest.raises(ManifestRefused) as e:
        mv.check_no_mixed_strata([pub, cl])
    assert e.value.reason == "mixed_tracks"
    nolabel = copy.deepcopy(pub)
    del nolabel["label"]
    with pytest.raises(ManifestRefused) as e:
        mv.check_no_mixed_strata([nolabel])
    assert e.value.reason == "missing_label"


def test_missing_fillers_flag(tmp_path):
    m = good(tmp_path)
    del m["utterances"][0]["fillers_in_ref"]
    assert reason(m) == "missing_fillers_flag"


def test_cli_exit_codes(tmp_path, capsys):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(good(tmp_path)))
    assert mv.main([str(p)]) == 0
    bad = good(tmp_path)
    del bad["label"]
    p.write_text(json.dumps(bad))
    assert mv.main([str(p)]) == 2
    assert "missing_label" in capsys.readouterr().err
    assert mv.main([str(tmp_path / "nope.json")]) == 2


# ---- DP-A (b): private corpus repo clone is the one allowed git tree ----

def _private_clone(tmp_path, monkeypatch, url="https://github.com/pestopoppa/epyc-stt-corpus-private.git"):
    root = tmp_path / "stt-corpus-private"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {url}\n')
    monkeypatch.setenv("STT_PRIVATE_CORPUS_ROOT", str(root))
    return root


def test_private_repo_clone_path_allowed(tmp_path, monkeypatch):
    root = _private_clone(tmp_path, monkeypatch)
    m = good(tmp_path, "private")
    (root / "s1" / "audio").mkdir(parents=True)
    (root / "s1" / "ref").mkdir(parents=True)
    m["audio_root"], m["reference_root"] = str(root / "s1" / "audio"), str(root / "s1" / "ref")
    mv.validate_manifest(m)


def test_public_tree_still_refused_when_private_clone_configured(tmp_path, monkeypatch):
    _private_clone(tmp_path, monkeypatch)
    m = good(tmp_path, "private")
    pub = tmp_path / "public-repo"
    (pub / ".git").mkdir(parents=True)
    (pub / "a").mkdir()
    m["audio_root"] = str(pub / "a")
    assert reason(m) == "private_path_in_git_tree"


def test_one_public_path_among_private_ones_refused(tmp_path, monkeypatch):
    root = _private_clone(tmp_path, monkeypatch)
    m = good(tmp_path, "private")
    (root / "a").mkdir()
    pub = tmp_path / "pub"
    (pub / ".git").mkdir(parents=True)
    m["audio_root"], m["reference_root"] = str(root / "a"), str(pub)
    assert reason(m) == "private_path_in_git_tree"


def test_clone_with_wrong_remote_not_trusted(tmp_path, monkeypatch):
    root = _private_clone(tmp_path, monkeypatch, url="https://github.com/pestopoppa/epyc-inference-research.git")
    m = good(tmp_path, "private")
    (root / "a").mkdir()
    m["audio_root"] = str(root / "a")
    assert reason(m) == "private_path_in_git_tree"


def test_nested_foreign_tree_inside_clone_refused(tmp_path, monkeypatch):
    root = _private_clone(tmp_path, monkeypatch)
    m = good(tmp_path, "private")
    (root / "sub" / ".git").mkdir(parents=True)
    m["audio_root"] = str(root / "sub")
    assert reason(m) == "private_path_in_git_tree"


def test_symlink_from_outside_into_public_tree_refused(tmp_path, monkeypatch):
    root = _private_clone(tmp_path, monkeypatch)
    pub = tmp_path / "pub"
    (pub / ".git").mkdir(parents=True)
    (pub / "a").mkdir()
    (root / "link").symlink_to(pub / "a")
    m = good(tmp_path, "private")
    m["audio_root"] = str(root / "link")
    assert reason(m) == "private_path_in_git_tree"
