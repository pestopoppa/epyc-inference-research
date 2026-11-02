import fcntl
import hashlib
import os

import pytest

from scripts.benchmark import kvu19a2_custody as custody


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_sealed_copy_is_immutable_and_requires_expected_source(tmp_path):
    source = tmp_path / "source.tar"
    source.write_bytes(b"trusted source")
    fd = custody.sealed_copy(source, digest(b"trusted source"))
    try:
        source.write_bytes(b"other source")
        assert os.read(fd, 100) == b"trusted source"
        assert fcntl.fcntl(fd, custody.F_GET_SEALS) & custody.SEAL_WRITE
        with pytest.raises(OSError):
            os.write(fd, b"tamper")
    finally:
        os.close(fd)
    with pytest.raises(ValueError, match="digest"):
        custody.sealed_copy(source, digest(b"trusted source"))


def test_process_sampler_refuses_coverage_and_digest_changes(tmp_path):
    pid = 123
    proc = tmp_path / str(pid)
    proc.mkdir()
    # Fields 3..21, then field 22 (start time).
    (proc / "stat").write_text("123 (weird ) command) " + " ".join(["0"] * 19 + ["42"]))
    (proc / "exe").write_bytes(b"exe")
    lib = tmp_path / "libggml-hip.so"
    lib.write_bytes(b"hip")
    st = lib.stat()
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
    (proc / "maps").write_text(f"1-2 r-xp 0 {dev} {st.st_ino} {lib}\n")
    result = custody.sample_process(pid, 42, digest(b"exe"), {str(lib): digest(b"hip")}, tmp_path)
    assert result["mapped_ggml"][str(lib)] == digest(b"hip")
    with pytest.raises(ValueError, match="birth"):
        custody.sample_process(pid, 43, digest(b"exe"), {str(lib): digest(b"hip")}, tmp_path)
    with pytest.raises(ValueError, match="digest"):
        custody.sample_process(pid, 42, digest(b"exe"), {str(lib): digest(b"wrong")}, tmp_path)
    (proc / "maps").write_text(f"1-2 r-xp 0 {dev} {st.st_ino} {lib} (deleted)\n")
    with pytest.raises(ValueError, match="unverifiable"):
        custody.sample_process(pid, 42, digest(b"exe"), {str(lib): digest(b"hip")}, tmp_path)
