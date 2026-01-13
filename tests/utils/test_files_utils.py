from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from capybara.utils import custom_path
from capybara.utils import files_utils as futils


def test_rm_path_removes_file_and_directory(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    assert f.exists()
    custom_path.rm_path(f)
    assert not f.exists()

    d = tmp_path / "empty_dir"
    d.mkdir()
    assert d.exists()
    custom_path.rm_path(d)
    assert not d.exists()


def test_copy_path_copies_and_validates_source(tmp_path: Path):
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    dst = tmp_path / "dst.txt"
    custom_path.copy_path(src, dst)
    assert dst.read_text(encoding="utf-8") == "hello"

    with pytest.raises(ValueError, match="invaild"):
        custom_path.copy_path(tmp_path / "missing.txt", dst)


def test_gen_md5_and_img_to_md5(tmp_path: Path):
    p = tmp_path / "blob.bin"
    payload = b"capybara"
    p.write_bytes(payload)

    assert futils.gen_md5(p) == hashlib.md5(payload).hexdigest()

    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert futils.img_to_md5(img) == hashlib.md5(img.tobytes()).hexdigest()

    with pytest.raises(TypeError, match="numpy array"):
        futils.img_to_md5("not-an-array")  # type: ignore[arg-type]


def test_dump_and_load_json_yaml_and_pickle(tmp_path: Path, monkeypatch):
    obj = {"a": 1, "b": [1, 2, 3]}

    json_path = tmp_path / "x.json"
    futils.dump_json(obj, json_path)
    assert futils.load_json(json_path) == obj

    yaml_path = tmp_path / "x.yaml"
    futils.dump_yaml(obj, yaml_path)
    assert futils.load_yaml(yaml_path) == obj

    pkl_path = tmp_path / "x.pkl"
    futils.dump_pickle(obj, pkl_path)
    assert futils.load_pickle(pkl_path) == obj

    # Default-path behavior should not pollute repo root.
    monkeypatch.chdir(tmp_path)
    futils.dump_json(obj, path=None)
    assert (tmp_path / "tmp.json").exists()
    futils.dump_yaml(obj, path=None)
    assert (tmp_path / "tmp.yaml").exists()


def test_get_files_filters_suffix_and_options(tmp_path: Path):
    # Create a few files with varying cases.
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.TXT").write_text("b", encoding="utf-8")
    (tmp_path / "c.jpg").write_text("c", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("d", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        futils.get_files(tmp_path / "missing")

    with pytest.raises(TypeError, match="suffix must be"):
        futils.get_files(tmp_path, suffix=123)  # type: ignore[arg-type]

    files = futils.get_files(tmp_path, suffix=".txt", recursive=True)
    assert all(Path(p).suffix.lower() == ".txt" for p in files)
    assert len(files) == 3

    files_no_case = futils.get_files(
        tmp_path,
        suffix=[".txt"],
        recursive=False,
        ignore_letter_case=False,
        sort_path=False,
        return_pathlib=False,
    )
    assert all(isinstance(p, str) for p in files_no_case)
    assert len(files_no_case) == 1
