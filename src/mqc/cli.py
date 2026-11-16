"""Command-line entry point.

mqc prepare --config configs/baseline.yaml   # download data, build QC dataset
mqc train   --config configs/baseline.yaml   # train and evaluate models
"""

import argparse
import json

from mqc import __version__
from mqc.config import load_config, set_seed
from mqc.data import download, load_images, md5sum
from mqc.dataset import build_qc_dataset, save_dataset, stratified_split
from mqc.train import train_models


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


def train(config: dict) -> None:
    if "models" not in config:
        raise ValueError("config has no 'models' section to train")
    run_dir = train_models(config)
    summary = json.loads((run_dir / "summary.json").read_text())
    print(f"Run saved to {run_dir}")
    print(
        f"{'model':<22}{'val acc':>9}{'test acc':>10}{'test 95% CI':>17}{'test F1':>9}"
    )
    for name, s in summary.items():
        val, test = s["val"], s["test"]
        lo, hi = test["accuracy_95ci"]
        print(
            f"{name:<22}{val['accuracy']:>9.3f}{test['accuracy']:>10.3f}"
            f"{f'[{lo:.3f}, {hi:.3f}]':>17}{test['macro_f1']:>9.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mqc", description="Microscopy image QC")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="download data and build QC dataset")
    p_prepare.add_argument("--config", default="configs/baseline.yaml")

    p_train = sub.add_parser("train", help="train and evaluate all models")
    p_train.add_argument("--config", default="configs/baseline.yaml")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    set_seed(config["seed"])

    if args.command == "prepare":
        prepare(config)
    elif args.command == "train":
        train(config)


if __name__ == "__main__":
    main()
