import json

import numpy as np
import yaml

from mqc import cli


def test_prepare_end_to_end(tmp_path, monkeypatch):
    """Run `mqc prepare` on a tiny fake dataset, without any network access."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw" / "fake.npz"
    raw.parent.mkdir()
    np.savez(raw, train_images=rng.integers(0, 256, (40, 28, 28, 3), dtype=np.uint8))
    monkeypatch.setattr(cli, "download", lambda name, raw_dir: raw)

    config = {
        "seed": 42,
        "data": {
            "dataset": "fake",
            "raw_dir": str(raw.parent),
            "processed_path": str(tmp_path / "processed" / "qc.npz"),
            "n_per_class": 5,
            "split": [0.6, 0.2, 0.2],
        },
        "qc_classes": ["good", "blurry", "overexposed", "noisy"],
        "degradations": {
            "blur_sigma": [0.8, 2.0],
            "exposure_gain": [1.5, 2.5],
            "noise_std": [0.05, 0.15],
        },
        "output_dir": str(tmp_path / "runs"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    cli.main(["prepare", "--config", str(config_path)])

    with np.load(tmp_path / "processed" / "qc.npz") as saved:
        assert saved["images"].shape == (20, 28, 28)
    meta = json.loads((tmp_path / "processed" / "qc.json").read_text())
    assert meta["config"]["seed"] == 42
    assert meta["split_sizes"] == {"train": 12, "val": 4, "test": 4}
