"""Moved: checkpoint initialization is init_checkpoint.py in the repository root. ``python -m gstar.init`` runs it."""


def main():
    from ._compat import run_script
    run_script("init_checkpoint.py")


if __name__ == "__main__":
    main()
