"""Durability and failure-boundary tests for the derived JSON publisher."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import status


class WriteJsonDurability(unittest.TestCase):

    def test_success_orders_file_sync_replace_and_directory_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = []
            file_fd = None
            directory_fd = None
            real_fsync = os.fsync
            real_open = os.open
            real_close = os.close
            real_replace = os.replace

            def fsync_spy(fd):
                nonlocal file_fd
                if fd == directory_fd:
                    events.append("directory fsync")
                else:
                    file_fd = fd
                    # Reading valid JSON here proves dump and flush preceded the
                    # file fsync, rather than merely checking call adjacency.
                    scratch, = root.glob(".durability-*")
                    self.assertEqual(json.loads(scratch.read_text()), {"value": 1})
                    events.append("file fsync")
                return real_fsync(fd)

            def replace_spy(source, target):
                events.append("replace")
                return real_replace(source, target)

            def open_spy(path, flags, *args, **kwargs):
                nonlocal directory_fd
                fd = real_open(path, flags, *args, **kwargs)
                if Path(path) == root:
                    directory_fd = fd
                    events.append("directory open")
                return fd

            def close_spy(fd):
                if fd == directory_fd:
                    events.append("directory close")
                return real_close(fd)

            with mock.patch.object(status.os, "fsync", side_effect=fsync_spy), \
                    mock.patch.object(status.os, "replace", side_effect=replace_spy), \
                    mock.patch.object(status.os, "open", side_effect=open_spy), \
                    mock.patch.object(status.os, "close", side_effect=close_spy):
                status.write_json(root, "derived.json", {"value": 1},
                                  prefix=".durability-")

            self.assertEqual(events, [
                "file fsync", "replace", "directory open",
                "directory fsync", "directory close",
            ])
            with self.assertRaises(OSError):
                os.fstat(file_fd)
            with self.assertRaises(OSError):
                os.fstat(directory_fd)

    def test_file_sync_failure_preserves_prior_target_and_closes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "derived.json"
            target.write_text('{"prior": true}', encoding="utf-8")
            unrelated = root / ".durability-unrelated"
            unrelated.write_text("belongs to somebody else", encoding="utf-8")
            created_fd = None
            real_mkstemp = tempfile.mkstemp

            def mkstemp_spy(*args, **kwargs):
                nonlocal created_fd
                created_fd, path = real_mkstemp(*args, **kwargs)
                return created_fd, path

            with mock.patch.object(status.tempfile, "mkstemp",
                                   side_effect=mkstemp_spy), \
                    mock.patch.object(status.os, "fsync",
                                      side_effect=OSError("file fsync failed")):
                with self.assertRaisesRegex(OSError, "file fsync failed"):
                    status.write_json(root, "derived.json", {"new": True},
                                      prefix=".durability-")

            self.assertEqual(target.read_text(encoding="utf-8"), '{"prior": true}')
            self.assertTrue(unrelated.is_file())
            self.assertEqual(list(root.glob(".durability-*")), [unrelated])
            with self.assertRaises(OSError):
                os.fstat(created_fd)

    def test_directory_sync_failure_is_reported_without_removing_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "derived.json").write_text('{"prior": true}', encoding="utf-8")
            calls = 0
            directory_fd = None
            real_fsync = os.fsync
            real_open = os.open

            def fsync_fault(fd):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("directory fsync failed")
                return real_fsync(fd)

            def open_spy(path, flags, *args, **kwargs):
                nonlocal directory_fd
                directory_fd = real_open(path, flags, *args, **kwargs)
                return directory_fd

            with mock.patch.object(status.os, "fsync", side_effect=fsync_fault), \
                    mock.patch.object(status.os, "open", side_effect=open_spy):
                with self.assertRaisesRegex(OSError, "directory fsync failed"):
                    status.write_json(root, "derived.json", {"new": True},
                                      prefix=".durability-")

            self.assertEqual(json.loads((root / "derived.json").read_text()),
                             {"new": True})
            self.assertEqual(list(root.glob(".durability-*")), [])
            with self.assertRaises(OSError):
                os.fstat(directory_fd)

    def test_fdopen_failure_closes_raw_descriptor_and_removes_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created_fd = None
            real_mkstemp = tempfile.mkstemp

            def mkstemp_spy(*args, **kwargs):
                nonlocal created_fd
                created_fd, path = real_mkstemp(*args, **kwargs)
                return created_fd, path

            with mock.patch.object(status.tempfile, "mkstemp",
                                   side_effect=mkstemp_spy), \
                    mock.patch.object(status.os, "fdopen",
                                      side_effect=OSError("fdopen failed")):
                with self.assertRaisesRegex(OSError, "fdopen failed"):
                    status.write_json(root, "derived.json", {"new": True},
                                      prefix=".durability-")

            self.assertEqual(list(root.glob(".durability-*")), [])
            with self.assertRaises(OSError):
                os.fstat(created_fd)

    def test_repeated_writes_publish_valid_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(12):
                expected = {"index": index, "payload": list(range(index))}
                returned = status.write_json(
                    root, "derived.json", expected, prefix=".durability-")
                self.assertEqual(returned, root / "derived.json")
                self.assertEqual(json.loads(returned.read_text(encoding="utf-8")),
                                 expected)
                self.assertEqual(list(root.glob(".durability-*")), [])


if __name__ == "__main__":
    unittest.main()
