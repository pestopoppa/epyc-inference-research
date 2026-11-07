import copy
from dataclasses import replace
import hashlib

import pytest

from .. import schemas
from . import longctx_c8 as cell
from . import serving


def _recipe():
    return serving.Recipe(name="c8", model="/model", np=8, ctx=270000,
                          device="ROCm0", ngl=99, n_predict=256)


def _texts():
    return [f"source-{i}" for i in range(8)]


def _encode(text):
    # Distinct deterministic synthetic token streams; no inference or tokenizer claim.
    return [int(text.rsplit("-", 1)[1])] * 32768


def test_roundtrip_binds_frozen_bytes_and_source():
    recipe, texts = _recipe(), _texts()
    manifest, proof = cell.build_manifest(recipe, texts, _encode, "a" * 64, "test")
    loaded = cell.validate_manifest(recipe, manifest, proof, texts, _encode, "a" * 64)
    assert len(loaded.prompts) == 8
    assert all(len(p.prompt) == 32768 and not p.cache_prompt for p in loaded.prompts)
    tampered = copy.deepcopy(manifest)
    request = tampered["prompts"][0]["request"]
    request["prompt"][0] = 100
    tampered["prompts"][0]["request_digest"] = hashlib.sha256(
        cell.ps._canonical(request)).hexdigest()
    body = {key: value for key, value in tampered.items() if key != "digest"}
    tampered["digest"] = schemas.content_hash(body)
    with pytest.raises(cell.WorkloadRefused):
        cell.validate_manifest(recipe, tampered, proof, texts, _encode, "a" * 64)
    changed_texts = list(texts)
    changed_texts[0] += " "
    with pytest.raises(cell.WorkloadRefused):
        cell.validate_manifest(recipe, manifest, proof, changed_texts, _encode, "a" * 64)
    with pytest.raises(cell.WorkloadRefused):
        cell.validate_manifest(recipe, manifest, proof, texts, _encode, "b" * 64)
    with pytest.raises(cell.WorkloadRefused):
        cell.validate_manifest(replace(recipe, temperature=0), manifest, proof, texts, _encode, "a" * 64)


def test_refuses_wrong_shape_and_capacity():
    recipe, texts = _recipe(), _texts()
    for invalid in (replace(recipe, np=4), replace(recipe, np=8.0),
                    replace(recipe, ngl=99.0), replace(recipe, device="none"),
                    replace(recipe, ctx=8*(32768+256+1)-1)):
        with pytest.raises(cell.WorkloadRefused):
            cell.build_manifest(invalid, texts, _encode, "a" * 64, "test")
    with pytest.raises(cell.WorkloadRefused, match="declare draft_n_max"):
        cell.build_manifest(replace(recipe, spec_decode={"type": "draft-mtp"}),
                            texts, _encode, "a" * 64, "test")
    with pytest.raises(cell.WorkloadRefused):
        cell.build_manifest(recipe, texts, lambda _: [0]*32768, "a"*64, "test")
    with pytest.raises(cell.WorkloadRefused):
        cell.build_manifest(recipe, texts, lambda _: [0]*30000, "a"*64, "test")
    with pytest.raises(cell.WorkloadRefused, match="tokenizer failed"):
        cell.build_manifest(recipe, texts, lambda _: None, "a"*64, "test")
