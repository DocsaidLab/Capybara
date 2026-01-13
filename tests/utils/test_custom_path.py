from __future__ import annotations

from capybara.utils.custom_path import rm_path


def test_rm_path_removes_non_empty_directories(tmp_path):
    folder = tmp_path / "nested"
    folder.mkdir()
    (folder / "file.txt").write_text("hello", encoding="utf-8")

    rm_path(folder)
    assert not folder.exists()


def test_rm_path_removes_files(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello", encoding="utf-8")

    rm_path(file_path)
    assert not file_path.exists()
