"""Explicit read-only binding for the historical control's private instrument.

Materialization is deliberately outside this module. Reopening reads the sealed
identity; only a new collection checks the live source/build before native work.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from .. import production_pins, schemas
from ..execution import t0_provider as tp

SCHEMA = "epyc.autokernel.historical_instrument.v1"
ENV = "AUTOKERNEL_HISTORICAL_INSTRUMENT"


class InstrumentBindingRefused(ValueError):
    pass


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout


def _compiler_version(path):
    result = subprocess.run([str(path), "--version"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    return result.stdout.decode().strip()


@dataclass(frozen=True)
class InstrumentBinding:
    """A sealed declaration, not authority to build or repair an instrument."""
    encoded: str

    @classmethod
    def from_mapping(cls, value):
        if type(value) is not dict or set(value) != {"schema", "source", "build", "binding_sha256"}:
            raise InstrumentBindingRefused("historical instrument declaration fields differ")
        body = {key: value[key] for key in ("schema", "source", "build")}
        if value["schema"] != SCHEMA or schemas.content_hash(body) != value["binding_sha256"]:
            raise InstrumentBindingRefused("historical instrument declaration digest differs")
        source, build = value["source"], value["build"]
        if type(source) is not dict or set(source) != {"root", "commit", "branch", "tree_oid", "ls_tree_sha256"}:
            raise InstrumentBindingRefused("historical instrument source fields differ")
        if type(build) is not dict or set(build) != {"root", "files", "compiler"}:
            raise InstrumentBindingRefused("historical instrument build fields differ")
        for name, length in (("tree_oid", 40), ("ls_tree_sha256", 64)):
            source_digest = source[name]
            if type(source_digest) is not str or len(source_digest) != length or any(c not in "0123456789abcdef" for c in source_digest):
                raise InstrumentBindingRefused("historical instrument source digest is malformed")
        compiler = build["compiler"]
        if type(compiler) is not dict or set(compiler) != {"path", "sha256", "version"}:
            raise InstrumentBindingRefused("historical instrument compiler identity is absent")
        if (type(compiler["path"]) is not str or not Path(compiler["path"]).is_absolute()
                or type(compiler["sha256"]) is not str or len(compiler["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in compiler["sha256"])
                or type(compiler["version"]) is not str or not compiler["version"].strip()):
            raise InstrumentBindingRefused("historical instrument compiler identity is malformed")
        if source["commit"] != production_pins.MEASUREMENT_COMMIT:
            raise InstrumentBindingRefused("historical instrument is not the original exact commit")
        if not isinstance(source["branch"], str) or not source["branch"].startswith(("experimental/", "llama.cpp-experimental")):
            raise InstrumentBindingRefused("historical instrument requires a declared experimental branch")
        for root in (source["root"], build["root"]):
            if type(root) is not str or not Path(root).is_absolute():
                raise InstrumentBindingRefused("historical instrument roots must be absolute")
        files = build["files"]
        if type(files) is not dict or not 4 <= len(files) <= 256:
            raise InstrumentBindingRefused("historical instrument file closure is absent or unbounded")
        required = {"CMakeCache.txt", *("bin/" + name for name in ("llama-bench", "llama-completion", "test-backend-ops"))}
        if not required <= files.keys():
            raise InstrumentBindingRefused("historical instrument tool closure is incomplete")
        for name, sha in files.items():
            if type(name) is not str:
                raise InstrumentBindingRefused("historical instrument file name is malformed")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or type(sha) is not str or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise InstrumentBindingRefused("historical instrument file identity is malformed")
        return cls(json.dumps(value, sort_keys=True, separators=(",", ":")))

    def to_dict(self):
        return json.loads(self.encoded)

    @property
    def source_root(self):
        return Path(self.to_dict()["source"]["root"])

    @property
    def build_root(self):
        return Path(self.to_dict()["build"]["root"])

    @property
    def branch(self):
        return self.to_dict()["source"]["branch"]

    @property
    def commit(self):
        return self.to_dict()["source"]["commit"]

    @property
    def source_sha256(self):
        return self.to_dict()["source"]["ls_tree_sha256"]

    @property
    def binary(self):
        return self.build_root / "bin" / "llama-bench"

    def validate_live(self):
        """Refuse drift before collection; never mutate/build or qualify replay."""
        source, build = self.source_root, self.build_root
        completion = self.binary.parent / "llama-completion"
        # Keep the actual frozen-tree guard, including symlink resolution.
        tp.CandidateBuild(str(source), str(build), self.commit, self.source_sha256,
            str(completion), str(completion.parent), str(completion.parent / "test-backend-ops"))
        if source.resolve() != source or build.resolve() != build:
            raise InstrumentBindingRefused("historical instrument source/build alias differs")
        data = self.to_dict()
        expected = data["source"]
        if (_git(source, "rev-parse", "HEAD").decode().strip() != self.commit
                or _git(source, "branch", "--show-current").decode().strip() != self.branch
                or _git(source, "status", "--porcelain", "--untracked-files=all", "--ignored=matching")
                or _git(source, "rev-parse", self.commit + "^{tree}").decode().strip() != expected["tree_oid"]
                or hashlib.sha256(_git(source, "ls-tree", "-r", "-z", self.commit)).hexdigest() != self.source_sha256):
            raise InstrumentBindingRefused("historical instrument Git source/branch/tree/clean binding differs")
        files = data["build"]["files"]
        actual_bin = {"bin/" + p.name for p in self.binary.parent.iterdir() if p.is_file()}
        if actual_bin != {name for name in files if name.startswith("bin/")}:
            raise InstrumentBindingRefused("historical instrument runtime file set differs")
        for name, expected_sha in files.items():
            path = build / name
            if not path.is_file() or not path.resolve().is_relative_to(build) or _sha(path) != expected_sha:
                raise InstrumentBindingRefused("historical instrument build/tool/DSO bytes differ: " + name)
        cache = (build / "CMakeCache.txt").read_text()
        homes = [line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
        if homes != [str(source)]:
            raise InstrumentBindingRefused("historical instrument CMake source binding differs")
        compiler = data["build"]["compiler"]
        cache_compilers = [line.split("=", 1)[1] for line in cache.splitlines()
            if line.startswith("CMAKE_CXX_COMPILER:")]
        expected_compiler = Path(compiler["path"])
        if (len(cache_compilers) != 1
                or Path(cache_compilers[0]).resolve() != expected_compiler.resolve()
                or not expected_compiler.is_file()
                or _sha(expected_compiler) != compiler["sha256"]
                or _compiler_version(expected_compiler) != compiler["version"]):
            raise InstrumentBindingRefused("historical instrument live compiler/cache identity differs")
        return self


def resolve(value=None):
    if type(value) is InstrumentBinding:
        return InstrumentBinding.from_mapping(value.to_dict())
    if value is not None:
        return InstrumentBinding.from_mapping(value)
    path = os.environ.get(ENV)
    if not path:
        return None
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 1024 * 1024:
        raise InstrumentBindingRefused("historical instrument manifest is missing/unsafe")
    return InstrumentBinding.from_mapping(json.loads(source.read_text()))
