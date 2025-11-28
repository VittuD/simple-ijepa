"""Thin shim preserving the old entrypoint name."""

from scripts.train_ddp import main


if __name__ == "__main__":
    main()
