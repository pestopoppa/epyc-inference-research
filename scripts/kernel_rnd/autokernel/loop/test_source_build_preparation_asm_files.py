"""Gap 1: only an isa_schedule asm_artifacts candidate may declare a bounded GPU-backend file set."""
from types import SimpleNamespace
import unittest

from autokernel.loop import source_build_preparation as sbp

FILES = ("ggml/src/ggml-cuda/fattn-asm-gfx90a.cu", "ggml/src/ggml-cuda/fattn.cu",
         "ggml/src/ggml-hip/CMakeLists.txt")


def m(files, cls="isa_schedule", arts=("co",)):
    return SimpleNamespace(declared_files=tuple(files), change_class=cls, asm_artifacts=arts)


class AsmMultiFileTests(unittest.TestCase):
    def test_single_file_unchanged(self):
        self.assertTrue(sbp._declared_files_admissible(m(["a.cu"], cls="kernel", arts=()), "a.cu"))
        self.assertFalse(sbp._declared_files_admissible(m(["a.cu"], cls="kernel", arts=()), "b.cu"))

    def test_asm_set_admitted(self):
        self.assertTrue(sbp._declared_files_admissible(m(FILES), FILES[0]))

    def test_refusals(self):
        cases = {
            "v1 multi-file": (m(FILES, arts=()), FILES[0]),
            "other class": (m(FILES, cls="kernel"), FILES[0]),
            "target absent": (m(FILES), "ggml/src/ggml-cuda/other.cu"),
            "outside backend": (m(FILES + ("tests/test-backend-ops.cpp",)), FILES[0]),
            "outside ggml-cpu": (m(FILES + ("ggml/src/ggml-cpu/x.c",)), FILES[0]),
            "too many": (m([f"ggml/src/ggml-cuda/f{i}.cu" for i in range(sbp.ASM_MULTI_FILE_MAX + 1)]),
                         "ggml/src/ggml-cuda/f0.cu"),
        }
        for label, (manifest, target) in cases.items():
            with self.subTest(label):
                self.assertFalse(sbp._declared_files_admissible(manifest, target))


if __name__ == "__main__":
    unittest.main()
