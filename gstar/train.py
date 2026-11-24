"""Moved: the training script is train.py in the repository root, the Trainer is gstar.trainer.ImageTrainer.
``python -m gstar.train`` runs train.py."""
from .data.collate import collate_attention_layout
from .trainer import ImageTrainer


def main():
    from ._compat import run_script
    run_script("train.py")


if __name__ == "__main__":
    main()
