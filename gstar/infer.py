"""Moved: the inference script is inference.py in the repository root. ``python -m gstar.infer`` runs it."""


def main():
    from ._compat import run_script
    run_script("inference.py")


if __name__ == "__main__":
    main()
