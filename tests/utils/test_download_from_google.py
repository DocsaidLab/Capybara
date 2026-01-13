from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import capybara.utils.utils as utils_mod


class _FakeResponse:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        text: str = "",
        chunks: list[bytes] | None = None,
        iter_raises: Exception | None = None,
    ) -> None:
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.text = text
        self._chunks = chunks or []
        self._iter_raises = iter_raises

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        if self._iter_raises is not None:
            raise self._iter_raises
        yield from self._chunks


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, stream=True):
        self.calls.append((url, params))
        if not self._responses:
            raise AssertionError("No more fake responses configured")
        return self._responses.pop(0)


class _DummyTqdm:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = kwargs.get("total", 0)
        self.updated = 0

    def update(self, n: int) -> None:
        self.updated += n

    def __enter__(self) -> _DummyTqdm:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _install_fakes(monkeypatch, session: _FakeSession) -> None:
    monkeypatch.setattr(utils_mod.requests, "Session", lambda: session)
    monkeypatch.setattr(utils_mod, "tqdm", _DummyTqdm)


def test_download_from_google_direct_content_disposition(tmp_path, monkeypatch):
    session = _FakeSession(
        [
            _FakeResponse(
                headers={
                    "content-disposition": "attachment; filename=x.bin",
                    "content-length": "3",
                },
                chunks=[b"abc"],
            )
        ]
    )
    _install_fakes(monkeypatch, session)

    out = utils_mod.download_from_google(
        file_id="id",
        file_name="x.bin",
        target=tmp_path,
    )
    assert out == Path(tmp_path) / "x.bin"
    assert out.read_bytes() == b"abc"
    assert session.calls[0][0] == "https://docs.google.com/uc"


def test_download_from_google_uses_cookie_confirm_token(tmp_path, monkeypatch):
    session = _FakeSession(
        [
            _FakeResponse(cookies={"download_warning_foo": "TOKEN"}, text=""),
            _FakeResponse(
                headers={
                    "content-disposition": "attachment",
                    "content-length": "1",
                },
                chunks=[b"x"],
            ),
        ]
    )
    _install_fakes(monkeypatch, session)

    out = utils_mod.download_from_google("id", "y.bin", target=tmp_path)
    assert out.read_bytes() == b"x"
    assert session.calls[1][1] is not None
    assert session.calls[1][1]["confirm"] == "TOKEN"


def test_download_from_google_parses_html_download_form(tmp_path, monkeypatch):
    html = """
    <html>
      <form id="download-form" action="https://example.com/download">
        <input type="hidden" name="id" value="X" />
        <input type="hidden" name="confirm" value="Y" />
      </form>
    </html>
    """
    session = _FakeSession(
        [
            _FakeResponse(text=html),
            _FakeResponse(
                headers={
                    "content-disposition": "attachment",
                    "content-length": "2",
                },
                chunks=[b"hi"],
            ),
        ]
    )
    _install_fakes(monkeypatch, session)

    out = utils_mod.download_from_google("id", "z.bin", target=tmp_path)
    assert out.read_bytes() == b"hi"
    assert session.calls[1][0] == "https://example.com/download"


def test_download_from_google_parses_confirm_param(tmp_path, monkeypatch):
    session = _FakeSession(
        [
            _FakeResponse(text="... confirm=ABC123 ..."),
            _FakeResponse(
                headers={
                    "content-disposition": "attachment",
                    "content-length": "1",
                },
                chunks=[b"1"],
            ),
        ]
    )
    _install_fakes(monkeypatch, session)

    out = utils_mod.download_from_google("id", "c.bin", target=tmp_path)
    assert out.read_bytes() == b"1"
    assert session.calls[1][1] is not None
    assert session.calls[1][1]["confirm"] == "ABC123"


def test_download_from_google_raises_when_no_link_found(tmp_path, monkeypatch):
    session = _FakeSession([_FakeResponse(text="<html />")])
    _install_fakes(monkeypatch, session)

    with pytest.raises(Exception, match="無法在回應中找到下載連結"):
        utils_mod.download_from_google("id", "x.bin", target=tmp_path)


def test_download_from_google_wraps_streaming_errors(tmp_path, monkeypatch):
    session = _FakeSession(
        [
            _FakeResponse(
                headers={
                    "content-disposition": "attachment",
                    "content-length": "1",
                },
                iter_raises=ValueError("boom"),
            )
        ]
    )
    _install_fakes(monkeypatch, session)

    with pytest.raises(RuntimeError, match="File download failed"):
        utils_mod.download_from_google("id", "x.bin", target=tmp_path)
