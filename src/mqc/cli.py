"""Command-line entry point.

mqc prepare --config configs/baseline.yaml   # download data, build QC dataset
mqc train   --config configs/baseline.yaml   # train and evaluate models
mqc report  --config configs/baseline.yaml   # calibration, review policy, figures
"""

import argparse
import json
import shutil
from pathlib import Path

from mqc import __version__
from mqc.config import load_config, set_seed
from mqc.data import download, load_images, md5sum
from mqc.dataset import build_qc_dataset, save_dataset, stratified_split
from mqc.report import build_report, latest_run
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


def report(config: dict, run: str | None, target_accuracy: float) -> None:
    run_dir = Path(run) if run else latest_run(config["output_dir"])
    out = build_report(run_dir, target_accuracy)
    res = json.loads((out / "report.json").read_text())
    print(f"Report saved to {out}")
    print(
        f"{'model':<22}{'temp':>6}{'ECE before':>12}{'ECE after':>11}"
        f"{'auto %':>8}{'auto acc':>10}{'errors caught':>15}"
    )
    for name, r in res["models"].items():
        rp = r["review_policy_test"]
        acc = "-" if rp["auto_accuracy"] is None else f"{rp['auto_accuracy']:.3f}"
        caught = f"{rp['errors_caught_by_review']}/{rp['total_errors']}"
        print(
            f"{name:<22}{r['temperature']:>6.2f}{r['test_ece_before']:>12.3f}"
            f"{r['test_ece_after']:>11.3f}{rp['auto_decided_frac']:>8.1%}"
            f"{acc:>10}{caught:>15}"
        )
    print(f"Recommended model: {res['recommended_model']}")

    # Copy figures to a stable, committed location so the README can show them.
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for png in out.glob("*.png"):
        shutil.copy2(png, figures_dir / png.name)
    print(f"Figures copied to {figures_dir}/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mqc", description="Microscopy image QC")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="download data and build QC dataset")
    p_prepare.add_argument("--config", default="configs/baseline.yaml")

    p_train = sub.add_parser("train", help="train and evaluate all models")
    p_train.add_argument("--config", default="configs/baseline.yaml")

    p_report = sub.add_parser("report", help="calibration, review policy, figures")
    p_report.add_argument("--config", default="configs/baseline.yaml")
    p_report.add_argument("--run", help="run folder (default: latest)")
    p_report.add_argument("--target-accuracy", type=float, default=0.99)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    set_seed(config["seed"])

    if args.command == "prepare":
        prepare(config)
    elif args.command == "train":
        train(config)
    elif args.command == "report":
        report(config, args.run, args.target_accuracy)


if __name__ == "__main__":
    main()
