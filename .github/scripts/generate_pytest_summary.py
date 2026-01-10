import os
import re
import textwrap
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

MARKER = "<!-- Pytest Report -->"
ARTIFACT_PLACEHOLDER = "<!-- PYTEST_ARTIFACT_LINK -->"
DEFAULT_LOG_TAIL_LINES = 200
DEFAULT_LOG_CHAR_LIMIT = 6000
SNIPPET_CHAR_LIMIT = 4000
TOP_SLOW_N = 20


@dataclass
class CoverageSummary:
    line_rate: float | None = None
    branch_rate: float | None = None
    line_pct: str | None = None
    branch_pct: str | None = None
    short_text: str = "覆蓋率資料不可用"


REPORT_DIR = Path(".ci-reports/pytest")
RAW_REPORT_DIR = Path(os.getenv("PYTEST_REPORT_DIR", ".ci-reports/pytest/raw"))
LEGACY_REPORT_DIR = Path(".pytest-reports")
COVERAGE_REPORT_FILE = "coverage_report.txt"


def _first_existing(paths: list[Path]) -> Path | None:
    for candidate in paths:
        if candidate.exists():
            return candidate
    return paths[0] if paths else None


def _as_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _threshold_value(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def fmt_outcome(value: str) -> str:
    mapping = {
        "success": "✅ 通過",
        "failure": "❌ 失敗",
        "cancelled": "⚪ 取消",
        "skipped": "⚪ 未執行",
    }
    value = (value or "").strip()
    return mapping.get(value, f"⚪ {value or '未知'}")


def fmt_percent(raw: str | float | int | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        value = float(raw)
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
    pct = value * 100.0
    if pct.is_integer():
        return f"{int(pct)}%"
    return f"{pct:.2f}%"


def cleanse_snippet(snippet: str, limit: int = SNIPPET_CHAR_LIMIT) -> str:
    snippet = textwrap.dedent(snippet).strip()
    if not snippet:
        return "(無訊息)"
    snippet = "\n".join(line.rstrip() for line in snippet.splitlines())
    if len(snippet) > limit:
        snippet = f"{snippet[:limit]}\n... (截斷)"
    return snippet


def load_log_tail(
    path: Path,
    *,
    limit: int = DEFAULT_LOG_CHAR_LIMIT,
    tail_lines: int = DEFAULT_LOG_TAIL_LINES,
) -> str | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if not raw:
        return None
    lines = raw.splitlines()
    tail = "\n".join(lines[-tail_lines:])
    if len(tail) > limit:
        tail = tail[-limit:]
    return tail


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def append_coverage_report(
    lines: list[str], report_path: Path, max_lines: int = 80
) -> None:
    raw = _read_text_if_exists(report_path)
    if raw is None:
        return
    content = [line.rstrip() for line in raw.splitlines() if line.strip()]
    if not content:
        return
    lines.append("")
    lines.append("### 覆蓋率缺口")
    lines.append("")
    lines.append("<details><summary>未覆蓋的檔案列表</summary>")
    lines.append("")
    lines.append("```")
    for line in content[:max_lines]:
        lines.append(line)
    if len(content) > max_lines:
        lines.append("... (截斷)")
    lines.append("```")
    lines.append("</details>")


def _iter_testcases(root: ElementTree.Element) -> list[ElementTree.Element]:
    return list(root.findall(".//testcase"))


def _parse_root(xml_path: Path) -> ElementTree.Element | None:
    if not xml_path.exists():
        return None
    try:
        return ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError:
        return None


def append_pytest_summary(lines: list[str], junit_path: Path) -> None:
    root = _parse_root(junit_path)
    if root is None:
        return

    attrs = root.attrib
    tests = attrs.get("tests")
    failures = attrs.get("failures") or attrs.get("failed")
    errors = attrs.get("errors")
    skipped = attrs.get("skipped") or attrs.get("disabled")
    time_spent = attrs.get("time")

    def _safe_int(x: str | int | float | None, default: int | str = 0) -> int:
        if x is None:
            return int(default)
        try:
            return int(float(x))
        except (TypeError, ValueError):
            try:
                return int(x)
            except (TypeError, ValueError):
                return int(default)

    if tests is None:
        tests = 0
        failures = 0
        errors = 0
        skipped = 0
        total_time = 0.0
        for suite in root.findall(".//testsuite"):
            a = suite.attrib
            tests += _safe_int(a.get("tests"), 0)
            failures += _safe_int(a.get("failures") or a.get("failed"), 0)
            errors += _safe_int(a.get("errors"), 0)
            skipped += _safe_int(a.get("skipped") or a.get("disabled"), 0)
            with suppress(Exception):
                total_time += float(a.get("time") or 0.0)
        time_spent = f"{total_time:.3f}"
    else:
        tests = _safe_int(tests, 0)
        failures = _safe_int(failures, 0)
        errors = _safe_int(errors, 0)
        skipped = _safe_int(skipped, 0)

    passed = max(tests - failures - errors - skipped, 0)

    lines.append(
        f"- 測試統計: 共 {tests}; ✅ 通過 {passed}; ❌ 失敗 {failures}; ⚠️ 錯誤 {errors}; ⏭️ 跳過 {skipped}; 耗時 {time_spent or 'N/A'} 秒"
    )

    failures_block: list[str] = []
    for case in _iter_testcases(root):
        failure = case.find("failure")
        error = case.find("error")
        target = failure or error
        if target is None:
            continue
        status = "failure" if failure is not None else "error"
        classname = case.attrib.get("classname") or ""
        name = case.attrib.get("name") or ""
        elapsed = case.attrib.get("time") or ""
        header = f"- `{classname}.{name}` ({status}"
        if elapsed:
            header += f", {elapsed} 秒"
        header += ")"
        snippet = target.attrib.get("message", "") or ""
        text_body = target.text or ""
        combined = "\n".join(part for part in (snippet, text_body) if part)
        failures_block.append(header)
        formatted = cleanse_snippet(combined)
        indented = "\n".join(f"  {line}" for line in formatted.splitlines())
        failures_block.append("  ```")
        failures_block.append(indented or "  (無訊息)")
        failures_block.append("  ```")

    if failures_block:
        lines.append("")
        lines.append("#### 失敗與錯誤測試")
        lines.append("")
        lines.extend(failures_block)


def append_top_slowest(
    lines: list[str], junit_path: Path, top_n: int = TOP_SLOW_N
) -> None:
    root = _parse_root(junit_path)
    if root is None:
        return
    records: list[tuple[float, str]] = []
    for case in _iter_testcases(root):
        try:
            t = float(case.attrib.get("time") or 0.0)
        except Exception:
            t = 0.0
        if t <= 0.0:
            continue
        classname = case.attrib.get("classname") or ""
        name = case.attrib.get("name") or ""
        fqname = f"{classname}.{name}" if classname else name
        records.append((t, fqname))
    if not records:
        return
    records.sort(key=lambda x: x[0], reverse=True)
    lines.append("")
    lines.append(f"#### 最慢測試 Top {min(top_n, len(records))}")
    lines.append("")
    lines.append("| 測試 | 耗時 (秒) |")
    lines.append("| --- | ---: |")
    for t, fqname in records[:top_n]:
        lines.append(f"| `{fqname}` | {t:.3f} |")


def append_warnings_summary(
    lines: list[str], candidate_logs: list[Path]
) -> None:
    text: str | None = None
    for p in candidate_logs:
        text = _read_text_if_exists(p)
        if text:
            break
    if not text:
        return

    start = None
    end = None
    all_lines = text.splitlines()
    for i, line in enumerate(all_lines):
        if re.search(r"=+\s*warnings summary\s*=+", line, re.IGNORECASE):
            start = i
            break
    if start is None:
        return
    for j in range(start + 1, len(all_lines)):
        if re.match(r"=+\s", all_lines[j]):
            end = j
            break
    block = (
        "\n".join(all_lines[start:end]).strip()
        if end is not None
        else "\n".join(all_lines[start:]).strip()
    )
    block = cleanse_snippet(block, limit=SNIPPET_CHAR_LIMIT)

    if block:
        lines.append("")
        lines.append("#### Warnings 摘要")
        lines.append("")
        lines.append("```")
        lines.append(block)
        lines.append("```")


def append_session_meta(lines: list[str], candidate_logs: list[Path]) -> None:
    text: str | None = None
    for p in candidate_logs:
        text = _read_text_if_exists(p)
        if text:
            break
    if not text:
        return
    platform_line = re.search(r"^platform .+$", text, re.MULTILINE)
    rootdir_line = re.search(r"^rootdir: .+$", text, re.MULTILINE)
    plugins_line = re.search(r"^plugins: .+$", text, re.MULTILINE)

    entries = []
    if platform_line:
        entries.append(f"- {platform_line.group(0)}")
    if rootdir_line:
        entries.append(f"- {rootdir_line.group(0)}")
    if plugins_line:
        entries.append(f"- {plugins_line.group(0)}")

    if entries:
        lines.append("")
        lines.append("### 測試環境")
        lines.append("")
        lines.extend(entries)


def append_coverage_summary(
    lines: list[str], coverage_path: Path
) -> CoverageSummary:
    summary = CoverageSummary()

    lines.append("")
    lines.append("### 覆蓋率")
    lines.append("")

    root = _parse_root(coverage_path)
    if root is None:
        lines.append("- 未產生覆蓋率報告 (可能因測試失敗或覆蓋率未啟用)。")
        summary.short_text = "覆蓋率資料不可用"
        return summary

    line_rate = _as_float(root.attrib.get("line-rate"))
    branch_rate = _as_float(root.attrib.get("branch-rate"))
    overall_line = fmt_percent(line_rate)
    overall_branch = fmt_percent(branch_rate)
    summary.line_rate = line_rate
    summary.branch_rate = branch_rate
    summary.line_pct = overall_line
    summary.branch_pct = overall_branch

    lines_valid = root.attrib.get("lines-valid")
    lines_covered = root.attrib.get("lines-covered")
    branches_valid = root.attrib.get("branches-valid")
    branches_covered = root.attrib.get("branches-covered")
    overall_parts: list[str] = []
    if overall_line:
        detail = overall_line
        if lines_covered and lines_valid:
            detail += f" ({lines_covered}/{lines_valid})"
        overall_parts.append(f"行 {detail}")
    if overall_branch:
        detail = overall_branch
        if branches_covered and branches_valid:
            detail += f" ({branches_covered}/{branches_valid})"
        overall_parts.append(f"分支 {detail}")
    if overall_parts:
        lines.append(f"- 總覆蓋率: {', '.join(overall_parts)}")
    else:
        lines.append("- 總覆蓋率資料不可用。")

    records: list[tuple[float, str, str, str]] = []
    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename") or cls.attrib.get("name")
        if not filename:
            continue
        line_rate = cls.attrib.get("line-rate")
        branch_rate = cls.attrib.get("branch-rate")
        try:
            line_value = float(line_rate) if line_rate is not None else 1.0
        except ValueError:
            line_value = 1.0
        records.append(
            (
                line_value,
                filename,
                fmt_percent(line_rate) or "N/A",
                fmt_percent(branch_rate) or "N/A",
            )
        )

    if records:
        records.sort(key=lambda item: item[0])
        lines.append("")
        lines.append("#### 覆蓋率最低的檔案 (前 10 名)")
        lines.append("")
        lines.append("| 檔案 | 行覆蓋 | 分支覆蓋 |")
        lines.append("| --- | --- | --- |")
        for _, filename, line_pct, branch_pct in records[:10]:
            lines.append(f"| `{filename}` | {line_pct} | {branch_pct} |")

    summary_parts: list[str] = []
    if overall_line:
        summary_parts.append(f"行 {overall_line}")
    if overall_branch:
        summary_parts.append(f"分支 {overall_branch}")
    summary.short_text = (
        " / ".join(summary_parts) if summary_parts else "覆蓋率資料不可用"
    )

    return summary


def build_coverage_kpi(
    info: CoverageSummary,
    min_line: float | None,
    min_branch: float | None,
) -> str:
    line_pct = info.line_pct
    branch_pct = info.branch_pct
    line_rate = info.line_rate
    branch_rate = info.branch_rate

    actual_parts: list[str] = []
    if line_pct:
        actual_parts.append(f"行 {line_pct}")
    if branch_pct:
        actual_parts.append(f"分支 {branch_pct}")
    actual_text = (
        " / ".join(actual_parts) if actual_parts else "覆蓋率資料不可用"
    )

    threshold_parts: list[str] = []
    if min_line is not None:
        min_line_pct = fmt_percent(min_line) or f"{min_line * 100:.2f}%"
        threshold_parts.append(f"行 {min_line_pct}")
    if min_branch is not None:
        min_branch_pct = fmt_percent(min_branch) or f"{min_branch * 100:.2f}%"
        threshold_parts.append(f"分支 {min_branch_pct}")
    threshold_text = ""
    if threshold_parts:
        threshold_text = f"(門檻 {' / '.join(threshold_parts)})"

    thresholds_provided = bool(threshold_parts)
    status_ok = True
    if min_line is not None and (
        line_rate is None or line_rate + 1e-9 < min_line
    ):
        status_ok = False
    if min_branch is not None and (
        branch_rate is None or branch_rate + 1e-9 < min_branch
    ):
        status_ok = False

    if info.short_text == "覆蓋率資料不可用":
        status_icon = "⚪"
    elif thresholds_provided:
        status_icon = "✅" if status_ok else "❌"
    else:
        status_icon = "✅"

    return f"{actual_text}{threshold_text} → {status_icon}"


def main() -> None:
    report_dir = REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "summary.md"

    outcome = os.getenv("PYTEST_OUTCOME", "")
    exit_code = os.getenv("PYTEST_EXIT_CODE", "")

    lines: list[str] = [MARKER, "### Pytest 結果"]
    status_line = f"- 狀態: {fmt_outcome(outcome)}"
    if exit_code:
        status_line += f" (exit={exit_code})"
    lines.append(status_line)

    junit_candidates = [
        RAW_REPORT_DIR / "pytest.xml",
        report_dir / "pytest.xml",
        LEGACY_REPORT_DIR / "pytest.xml",
    ]
    coverage_candidates = [
        RAW_REPORT_DIR / "coverage.xml",
        report_dir / "coverage.xml",
        LEGACY_REPORT_DIR / "coverage.xml",
    ]
    junit_path = _first_existing(junit_candidates) or junit_candidates[0]
    coverage_xml = (
        _first_existing(coverage_candidates) or coverage_candidates[0]
    )
    log_candidates = [
        report_dir / "pytest.log",
        report_dir / "run.log",
        RAW_REPORT_DIR / "pytest.log",
        LEGACY_REPORT_DIR / "pytest.log",
    ]

    append_session_meta(lines, log_candidates)
    append_pytest_summary(lines, junit_path)
    append_top_slowest(lines, junit_path, top_n=TOP_SLOW_N)
    append_warnings_summary(lines, log_candidates)
    coverage_info = append_coverage_summary(lines, coverage_xml)
    append_coverage_report(lines, report_dir / COVERAGE_REPORT_FILE)

    min_line = _threshold_value(os.getenv("COVERAGE_MIN_LINE"))
    min_branch = _threshold_value(os.getenv("COVERAGE_MIN_BRANCH"))
    coverage_kpi = build_coverage_kpi(coverage_info, min_line, min_branch)

    lines.append("")
    lines.append(f"- 產物: {ARTIFACT_PLACEHOLDER}")

    if outcome.strip() == "failure":
        tail = None
        for p in log_candidates:
            tail = load_log_tail(p)
            if tail:
                break
        if tail:
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Pytest 輸出 (最後 200 行)</summary>")
            lines.append("")
            lines.append("```")
            lines.append(tail)
            lines.append("```")
            lines.append("</details>")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    coverage_txt_path = report_dir / "coverage.txt"
    coverage_txt_path.write_text(f"{coverage_kpi}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
