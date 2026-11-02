"""Original-process custody for KVU-12.6. Never infer identity from a build path."""
import hashlib
import json
import os
import re
import time
from pathlib import Path


class IdentityError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def birth(pid):
    # comm can contain spaces and parentheses; fields following the last ')' start at field 3.
    text = Path(f"/proc/{pid}/stat").read_text()
    return text.rsplit(")", 1)[1].split()[19]


def mapped_libraries(pid):
    result = {}
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if not re.search(r"/lib(?:ggml|llama|mtmd)[^/]*\.so(?:\.|$)", path):
            continue
        if path.endswith(" (deleted)"):
            raise IdentityError("mapped runtime library was deleted")
        file = Path(path)
        st = file.stat()
        major, minor = (int(x, 16) for x in fields[3].split(":"))
        if (int(fields[4]) != st.st_ino or major != os.major(st.st_dev) or minor != os.minor(st.st_dev)):
            raise IdentityError(f"mapped runtime library identity changed: {file}")
        basename = file.name
        if basename in result and result[basename][0] != str(file):
            raise IdentityError(f"duplicate runtime library basename: {basename}")
        hashed = sha256_file(file)
        if stat_signature(file.stat()) != stat_signature(st):
            raise IdentityError(f"runtime library changed while hashing: {file}")
        result[basename] = (str(file), hashed)
    if not any(x.startswith("libllama") for x in result) or not any(x.startswith("libggml") for x in result):
        raise IdentityError("missing mapped llama or ggml library")
    return result


class RuntimeVerifier:
    """Check model, executable, mapped DSOs and birth before and after a wave.

    Model digest is computed before server creation; an open fd and unchanged file
    identity detect replacement/mutation. The model must also be the actual -m
    argument in the running server. This does not prove server-internal model
    state or authenticate an independently supplied build receipt.
    """
    def __init__(self, model, expected_model_sha256, build):
        if build.get("schema") != "epyc.kvu126.build_identity.v1" or build.get("source_tree_clean") is not True:
            raise IdentityError("missing or invalid build receipt")
        if not build.get("binary_sha256") or not build.get("dso_sha256"):
            raise IdentityError("build receipt lacks executable or DSO hashes")
        self.build = build
        self.model = Path(model).resolve(strict=True)
        self.model_fd = open(self.model, "rb")
        self.model_stat = stat_signature(os.fstat(self.model_fd.fileno()))
        self.model_sha256 = sha256_file(self.model)
        if self.model_sha256 != expected_model_sha256:
            self.close()
            raise IdentityError("model digest differs from predeclaration")

    def close(self):
        self.model_fd.close()

    def check(self, pid):
        try:
            before = birth(pid)
            stat_open = stat_signature(os.fstat(self.model_fd.fileno()))
            stat_path = stat_signature(self.model.stat())
            if stat_open != self.model_stat or stat_path != self.model_stat:
                raise IdentityError("model file was replaced or mutated")
            exe = Path(os.readlink(f"/proc/{pid}/exe"))
            if str(exe).endswith(" (deleted)"):
                raise IdentityError("mapped executable was deleted")
            st = exe.stat()
            proc_exe_st = os.stat(f"/proc/{pid}/exe")
            if stat_signature(st) != stat_signature(proc_exe_st):
                raise IdentityError("executable path no longer matches mapped executable")
            binary_sha256 = sha256_file(exe)
            if stat_signature(exe.stat()) != stat_signature(st):
                raise IdentityError("executable changed while hashing")
            if binary_sha256 != self.build["binary_sha256"]:
                raise IdentityError("running executable differs from build receipt")
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            decoded = [os.fsdecode(x) for x in argv if x]
            model_flags = [decoded[i + 1] for i, value in enumerate(decoded[:-1]) if value in ("-m", "--model")]
            if len(model_flags) != 1 or Path(model_flags[0]).resolve() != self.model:
                raise IdentityError("running server model argument differs")
            libs = mapped_libraries(pid)
            lib_hashes = {name: value[1] for name, value in libs.items()}
            if lib_hashes != self.build["dso_sha256"]:
                raise IdentityError("mapped runtime libraries differ from build receipt")
            if birth(pid) != before:
                raise IdentityError("server process identity changed during snapshot")
            return {"pid_birth_identity": f"{pid}:{before}", "binary_sha256": binary_sha256,
                    "model_sha256": self.model_sha256, "mapped_dso_sha256": lib_hashes,
                    "mapped_dso_paths": {name: value[0] for name, value in libs.items()},
                    "binary_stat": stat_signature(proc_exe_st),
                    "mapped_dso_stats": {name: stat_signature(Path(value[0]).stat()) for name, value in libs.items()},
                    "captured_monotonic_s": time.monotonic()}
        except (OSError, ValueError, IndexError) as exc:
            raise IdentityError(f"runtime identity unavailable: {exc}") from exc

    def fast_check(self, pid, snapshot):
        """Check the same process and mapped files around each HTTP sample."""
        try:
            if f"{pid}:{birth(pid)}" != snapshot["pid_birth_identity"]:
                raise IdentityError("server birth identity changed")
            if (stat_signature(self.model.stat()) != self.model_stat or
                    stat_signature(os.fstat(self.model_fd.fileno())) != self.model_stat):
                raise IdentityError("model identity changed")
            if stat_signature(os.stat(f"/proc/{pid}/exe")) != tuple(snapshot["binary_stat"]):
                raise IdentityError("running executable identity changed")
            maps = Path(f"/proc/{pid}/maps").read_text()
            mapped = {}
            for line in maps.splitlines():
                fields = line.split(None, 5)
                if len(fields) < 6 or not re.search(r"/lib(?:ggml|llama|mtmd)[^/]*\.so(?:\.|$)", fields[5]):
                    continue
                path = fields[5]
                if path.endswith(" (deleted)"):
                    raise IdentityError("mapped runtime library was deleted")
                st = Path(path).stat()
                major, minor = (int(x, 16) for x in fields[3].split(":"))
                if (int(fields[4]), major, minor) != (st.st_ino, os.major(st.st_dev), os.minor(st.st_dev)):
                    raise IdentityError("mapped runtime library identity changed")
                mapped[Path(path).name] = stat_signature(st)
            if mapped != {name: tuple(value) for name, value in snapshot["mapped_dso_stats"].items()}:
                raise IdentityError("mapped runtime libraries changed")
            return snapshot["pid_birth_identity"]
        except (OSError, ValueError, IndexError) as exc:
            raise IdentityError(f"per-sample identity unavailable: {exc}") from exc


def verify_producer_pins(expected):
    names = ("kvu_cpu_matrix.py", "kvu126_capture.py", "kvu126_identity.py", "kvu126_compare.py", "kvu126_build.py", "kvu126_finalize.py")
    if not isinstance(expected, dict) or set(expected) != set(names):
        raise IdentityError("predeclared producer pins missing or incomplete")
    actual = {name: sha256_file(Path(__file__).resolve().parent / name) for name in names}
    if actual != expected:
        raise IdentityError("producer sources changed or do not match the predeclared pins")
    return actual


def write_runtime_receipt(directory, wave, row_sha256, before, after, plan_sha256, producer_pins):
    keys = ("pid_birth_identity", "binary_sha256", "model_sha256", "mapped_dso_sha256")
    if any(before[key] != after[key] for key in keys):
        raise IdentityError("runtime identity changed over the wave")
    receipt = {"schema": "epyc.kvu126.runtime_identity.v1", "row_sha256": row_sha256, "plan_sha256": plan_sha256,
               "producer_pins": producer_pins,
               **{key: before[key] for key in keys}, "before": before, "after": after}
    path = Path(directory) / f"wave-{wave}-runtime.json"
    with path.open("x") as out:
        json.dump(receipt, out, indent=2)
        out.write("\n")
    return path
