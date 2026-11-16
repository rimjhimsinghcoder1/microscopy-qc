"""Command-line entry point.

mqc prepare --config configs/baseline.yaml   # download data, build QC dataset
"""

import argparse

from mqc import __version__
from mqc.config import load_config, set_seed
from mqc.data import download, load_images, md5sum
from mqc.dataset import build_qc_dataset, save_dataset, stratified_split


def prepare(config: dict) -> None:
    data_cfg = config["data"]
    raw_path = download(data_cfg["dataset"], data_cfg["raw_dir"])
    images = load_images(raw_path)
    print(f"Loaded {len(images)} source images of shape {images.shape[1:]}")

    dataset = build_qc_dataset(
        images,
        config["qc_classes"],
        data_cfg["n_per_class"],
        config["degradations"],
        config["seed"],
    )
    splits = stratified_split(dataset["labels"], data_cfg["split"], config["seed"])
    out = save_dataset(
        data_cfg["processed_path"],
        dataset,
        splits,
        metadata={
            "config": config,
            "source_file": raw_path.name,
            "source_md5": md5sum(raw_path),
        },
    )
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"Saved {len(dataset['labels'])} images to {out} | splits: {sizes}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mqc", description="Microscopy image QC")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="download data and build QC dataset")
    p_prepare.add_argument("--config", default="configs/baseline.yaml")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    set_seed(config["seed"])

    if args.command == "prepare":
        prepare(config)


if __name__ == "__main__":
    main()
