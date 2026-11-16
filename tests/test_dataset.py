import json

import numpy as np
import pytest

from mqc.dataset import build_qc_dataset, save_dataset, stratified_split

CLASSES = ["good", "blurry", "overexposed", "noisy"]


def test_dataset_is_balanced(image_stack, ranges):
    ds = build_qc_dataset(image_stack, CLASSES, 8, ranges, seed=0)
    assert ds["images"].shape == (32, 28, 28)
    assert np.bincount(ds["labels"]).tolist() == [8, 8, 8, 8]


def test_each_source_image_used_once(image_stack, ranges):
    ds = build_qc_dataset(image_stack, CLASSES, 8, ranges, seed=0)
    assert len(np.unique(ds["source_index"])) == len(ds["source_index"])


def test_same_seed_same_dataset(image_stack, ranges):
    a = build_qc_dataset(image_stack, CLASSES, 8, ranges, seed=3)
    b = build_qc_dataset(image_stack, CLASSES, 8, ranges, seed=3)
    for key in a:
        assert np.array_equal(a[key], b[key], equal_nan=True)


def test_different_seed_different_dataset(image_stack, ranges):
    a = build_qc_dataset(image_stack, CLASSES, 8, ranges, seed=3)
    b = build_qc_dataset(image_stack, CLASSES, 8, ranges, seed=4)
    assert not np.array_equal(a["source_index"], b["source_index"])


def test_too_few_source_images_raises(image_stack, ranges):
    with pytest.raises(ValueError, match="source images"):
        build_qc_dataset(image_stack, CLASSES, 11, ranges, seed=0)


def test_split_has_no_overlap_and_covers_everything():
    labels = np.repeat(np.arange(4), 20)
    splits = stratified_split(labels, [0.7, 0.15, 0.15], seed=0)
    combined = np.concatenate(list(splits.values()))
    assert len(combined) == len(labels)
    assert len(np.unique(combined)) == len(labels)


def test_split_keeps_classes_balanced():
    labels = np.repeat(np.arange(4), 20)
    splits = stratified_split(labels, [0.7, 0.15, 0.15], seed=0)
    assert np.bincount(labels[splits["train"]]).tolist() == [14, 14, 14, 14]
    assert np.bincount(labels[splits["test"]]).tolist() == [3, 3, 3, 3]


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        stratified_split(np.zeros(10, dtype=int), [0.5, 0.5, 0.5], seed=0)


def test_save_writes_arrays_and_metadata(tmp_path, image_stack, ranges):
    ds = build_qc_dataset(image_stack, CLASSES, 8, ranges, seed=0)
    splits = stratified_split(ds["labels"], [0.5, 0.25, 0.25], seed=0)
    path = save_dataset(tmp_path / "qc.npz", ds, splits, {"seed": 0})

    with np.load(path) as saved:
        assert np.array_equal(saved["labels"], ds["labels"])
        assert np.array_equal(saved["split_train"], splits["train"])

    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["seed"] == 0
    assert meta["n_samples"] == 32
    assert "created_utc" in meta
