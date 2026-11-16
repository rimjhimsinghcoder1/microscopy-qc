import numpy as np
import pytest

from mqc import data


def test_grayscale_conversion_range_and_shape():
    rgb = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    gray = data.to_grayscale(rgb)
    assert gray.shape == (2, 4, 4)
    assert np.allclose(gray, 1.0)


def test_md5_matches_known_value(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_bytes(b"hello")
    assert data.md5sum(f) == "5d41402abc4b2a76b9719d911017c592"


def test_existing_valid_file_is_not_downloaded_again(tmp_path, monkeypatch):
    f = tmp_path / "fake.npz"
    f.write_bytes(b"hello")
    monkeypatch.setitem(
        data.DATASETS,
        "fake",
        {"url": "http://example.invalid", "md5": data.md5sum(f)},
    )

    def fail(*args, **kwargs):
        raise AssertionError("download should have been skipped")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fail)
    assert data.download("fake", tmp_path) == f


def test_checksum_mismatch_raises_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setitem(
        data.DATASETS, "fake", {"url": "http://example.invalid", "md5": "0" * 32}
    )
    monkeypatch.setattr(
        data.urllib.request,
        "urlretrieve",
        lambda url, dest: dest.write_bytes(b"corrupted"),
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data.download("fake", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        data.download("nope", tmp_path)
