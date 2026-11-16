"""Command-line entry point: `mqc --config configs/baseline.yaml`."""

import argparse

from mqc import __version__
from mqc.config import load_config, set_seed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mqc", description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    set_seed(config["seed"])
    print(f"mqc {__version__} | seed={config['seed']} | classes={config['qc_classes']}")


if __name__ == "__main__":
    main()
