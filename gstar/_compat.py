"""Entry points of earlier versions (``python -m gstar.train`` / ``gstar.infer`` / ``gstar.init``) run the scripts in
the repository root."""
import runpy
import sys

from .utils.config import ROOT


def run_script(name):
    path = ROOT / name
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name="__main__")
