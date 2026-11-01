"""Baseline source isolation for one OAB-31 control test."""
import importlib.abc
import importlib.util
import subprocess
import sys

REV = "94b1dd64c163d915eb1347b118f57e44e981bf78"
REPO = "/mnt/micron/codex3-lanes/research"
NAMES = {"actors", "bestof", "loop", "pipeline", "pool", "resume", "run",
         "test_runtime_recovery"}

class BaselineLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname.startswith("autokernel.loop.") or
                fullname.startswith("scripts.kernel_rnd.autokernel.loop.")):
            return None
        name = fullname.rsplit(".", 1)[-1]
        if name not in NAMES:
            return None
        return importlib.util.spec_from_loader(fullname, self, origin=(
            REPO + "/scripts/kernel_rnd/autokernel/loop/" + name + ".py"))
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        name = module.__name__.rsplit(".", 1)[-1]
        rel = "scripts/kernel_rnd/autokernel/loop/" + name + ".py"
        raw = subprocess.check_output(["git", "-C", REPO, "show", REV + ":" + rel])
        module.__file__ = REPO + "/" + rel
        print("BASELINE-OBJECT", module.__name__, REV, file=sys.stderr, flush=True)
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
    def get_code(self, fullname):
        name = fullname.rsplit(".", 1)[-1]
        rel = "scripts/kernel_rnd/autokernel/loop/" + name + ".py"
        raw = subprocess.check_output(["git", "-C", REPO, "show", REV + ":" + rel])
        print("BASELINE-OBJECT-CODE", fullname, REV, file=sys.stderr, flush=True)
        return compile(raw, REPO + "/" + rel, "exec")

sys.meta_path.insert(0, BaselineLoader())
