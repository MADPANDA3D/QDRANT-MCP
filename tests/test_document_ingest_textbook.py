import subprocess
import sys
import types
from pathlib import Path

import pytest

import mcp_server_qdrant.document_ingest as document_ingest
from mcp_server_qdrant.document_ingest import (
    DocumentSection,
    compute_ocr_page_budget,
    detect_pdf_chapter_markers,
    normalize_chapter_map,
    normalize_text_for_chunking,
    resolve_chapter_metadata_for_page,
    select_ocr_candidate_pages,
    summarize_pdf_text_coverage,
)
from mcp_server_qdrant.settings import MemorySettings


def test_normalize_text_for_chunking_collapses_whitespace():
    raw = "Line 1   with   spaces\r\n\r\n\r\nLine 2\t\ttext"
    normalized = normalize_text_for_chunking(raw)
    assert normalized == "Line 1 with spaces\n\nLine 2 text"


def test_normalize_chapter_map_valid_and_sorted():
    chapter_map = normalize_chapter_map(
        [
            {"start_page": 20, "end_page": 40, "chapter": 2, "chapter_title": "B"},
            {"start_page": 1, "end_page": 19, "chapter": 1, "chapter_title": "A"},
        ]
    )
    assert chapter_map[0]["start_page"] == 1
    assert chapter_map[1]["start_page"] == 20


def test_normalize_chapter_map_rejects_invalid_start_page():
    with pytest.raises(ValueError):
        normalize_chapter_map([{"start_page": 0, "chapter": 1}])


def test_detect_pdf_chapter_markers_and_resolve():
    sections = [
        DocumentSection(
            text="CHAPTER 1: Foundations\nBody text here",
            page_start=1,
            page_end=1,
        ),
        DocumentSection(
            text="Some intermediate page text",
            page_start=2,
            page_end=2,
        ),
        DocumentSection(
            text="Chapter 2 - Communication in Context\nMore text",
            page_start=10,
            page_end=10,
        ),
    ]
    markers = detect_pdf_chapter_markers(sections)
    assert len(markers) == 2
    assert markers[0]["chapter"] == 1
    assert markers[1]["chapter"] == 2

    chapter, title = resolve_chapter_metadata_for_page(
        9, detected_markers=markers, chapter_map=None
    )
    assert chapter == 1
    assert title == "Foundations"

    chapter_2, title_2 = resolve_chapter_metadata_for_page(
        10, detected_markers=markers, chapter_map=None
    )
    assert chapter_2 == 2
    assert title_2 == "Communication in Context"


def test_chapter_map_overrides_detected_markers():
    markers = [
        {"start_page": 1, "chapter": 1, "chapter_title": "Detected 1"},
        {"start_page": 10, "chapter": 2, "chapter_title": "Detected 2"},
    ]
    chapter_map = normalize_chapter_map(
        [{"start_page": 5, "end_page": 8, "chapter": 7, "chapter_title": "Override"}]
    )
    chapter, title = resolve_chapter_metadata_for_page(
        6, chapter_map=chapter_map, detected_markers=markers
    )
    assert chapter == 7
    assert title == "Override"


def test_coverage_summary_counts_low_text_pages():
    page_texts = [
        "This page has enough text to be considered extracted.",
        "",
        "tiny",
        "Another page with healthy text output.",
    ]
    summary = summarize_pdf_text_coverage(page_texts, low_text_threshold_chars=10)
    assert summary["total_pages"] == 4
    assert summary["good_pages"] == 2
    assert summary["low_text_pages"] == [1, 2]
    assert summary["coverage_ratio"] == 0.5


def test_compute_ocr_budget_respects_ratio_and_cap():
    budget = compute_ocr_page_budget(434, ocr_max_pages=120, ocr_max_page_ratio=0.30)
    assert budget == 120

    tiny_budget = compute_ocr_page_budget(3, ocr_max_pages=120, ocr_max_page_ratio=0.30)
    assert tiny_budget == 1


def test_select_ocr_candidates_prefers_lowest_text_pages():
    page_texts = [
        "text-rich page with enough characters",
        "x",
        "",
        "short",
        "another good page with enough characters",
    ]
    candidates = select_ocr_candidate_pages(
        page_texts,
        low_text_threshold_chars=8,
        budget_pages=2,
    )
    assert candidates == [2, 1]


def test_textbook_page_limit_accepts_large_full_ingest_default():
    settings = MemorySettings()

    assert document_ingest.textbook_page_limit_exceeded(3526, settings.textbook_max_pages) is False
    assert document_ingest.textbook_page_limit_exceeded(5001, settings.textbook_max_pages) is True


def test_pdf_preflight_counts_pages(tmp_path):
    from pypdf import PdfWriter

    pdf_path = tmp_path / "three-pages.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    assert document_ingest.get_pdf_page_count(pdf_path) == 3


def test_antiword_timeout_is_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == document_ingest.DOCUMENT_SUBPROCESS_TIMEOUT_SECONDS
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert not kwargs.get("shell", False)
        if document_ingest._resource is not None:  # pylint: disable=protected-access
            assert callable(kwargs["preexec_fn"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(document_ingest.subprocess, "run", fake_run)
    result = document_ingest._extract_doc_sections_sync(b"synthetic-doc")

    assert result.sections == []
    assert "antiword timed out" in " ".join(result.warnings)


def test_antiword_output_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        if document_ingest._resource is not None:  # pylint: disable=protected-access
            assert callable(kwargs["preexec_fn"])
        kwargs["stdout"].write(b"x" * 11)
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(document_ingest.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="configured size limit"):
        document_ingest._extract_doc_sections_sync(
            b"synthetic-doc",
            max_extracted_chars=10,
        )


def test_subprocess_file_size_limiter_sets_child_rlimit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource = types.SimpleNamespace(
        RLIMIT_FSIZE=1,
        RLIM_INFINITY=-1,
        getrlimit=lambda _resource_id: (-1, -1),
        setrlimit=lambda resource_id, limits: calls.append((resource_id, limits)),
    )
    monkeypatch.setattr(document_ingest, "_resource", fake_resource)

    limiter = document_ingest._subprocess_file_size_limiter(4096)
    assert limiter is not None
    limiter()

    assert calls == [(fake_resource.RLIMIT_FSIZE, (4096, -1))]


def test_pdftotext_timeout_does_not_expose_subprocess_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_private_path = "/" + "home/example/private.pdf"
    secret = f"Bearer test-secret {synthetic_private_path}"

    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == document_ingest.DOCUMENT_SUBPROCESS_TIMEOUT_SECONDS
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        if document_ingest._resource is not None:  # pylint: disable=protected-access
            assert callable(kwargs["preexec_fn"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], stderr=secret)

    monkeypatch.setattr(document_ingest.subprocess, "run", fake_run)
    text, warnings = document_ingest._extract_pdf_text_with_pdftotext(b"synthetic-pdf")

    serialized = " ".join(warnings)
    assert text == ""
    assert "pdftotext timed out" in serialized
    assert "test-secret" not in serialized
    assert synthetic_private_path.rsplit("/", 1)[0] not in serialized


def test_pdftotext_output_is_bounded_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args, **kwargs):
        if document_ingest._resource is not None:  # pylint: disable=protected-access
            assert callable(kwargs["preexec_fn"])
        Path(args[-1]).write_bytes(b"x" * 11)
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(document_ingest.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="configured size limit"):
        document_ingest._extract_pdf_text_with_pdftotext(
            b"synthetic-pdf",
            max_output_bytes=10,
        )


def test_pdf_page_limit_is_checked_before_page_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = False

    class FakePage:
        def extract_text(self) -> str:
            nonlocal extracted
            extracted = True
            return "should not run"

    class FakeReader:
        def __init__(self, *_args, **_kwargs) -> None:
            self.pages = [FakePage(), FakePage()]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakeReader))
    with pytest.raises(ValueError, match="page count"):
        document_ingest._extract_pdf_sections_sync(
            b"synthetic-pdf",
            ocr=False,
            max_pages=1,
        )
    assert extracted is False


def test_pdf_and_ocr_errors_are_redacted_and_ocr_has_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_private_path = "/" + "home/example/file.pdf"
    secret = f"Bearer test-secret https://private.example/token {synthetic_private_path}"

    class FailingPage:
        def extract_text(self) -> str:
            raise RuntimeError(secret)

    class EmptyPage:
        def extract_text(self) -> str:
            return ""

    class FakeReader:
        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            type(self).calls += 1
            self.pages = [FailingPage()] if self.calls == 1 else [EmptyPage()]

    class FakeImage:
        def close(self) -> None:
            return None

    def fake_convert(*_args, **kwargs):
        assert kwargs["timeout"] == document_ingest.DOCUMENT_SUBPROCESS_TIMEOUT_SECONDS
        assert kwargs["dpi"] == document_ingest.OCR_RENDER_DPI
        assert kwargs["size"] == document_ingest.OCR_RENDER_MAX_DIMENSION
        assert kwargs["thread_count"] == 1
        return [FakeImage()]

    def fake_ocr(*_args, **kwargs):
        assert kwargs["timeout"] == document_ingest.OCR_PAGE_TIMEOUT_SECONDS
        raise RuntimeError(secret)

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakeReader))
    monkeypatch.setitem(
        sys.modules,
        "pdf2image",
        types.SimpleNamespace(
            convert_from_bytes=fake_convert,
            convert_from_path=fake_convert,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        types.SimpleNamespace(image_to_string=fake_ocr),
    )
    monkeypatch.setattr(
        document_ingest,
        "_extract_pdf_text_with_pdftotext",
        lambda *_args, **_kwargs: ("", []),
    )

    parser_result = document_ingest._extract_pdf_sections_sync(
        b"synthetic-pdf",
        ocr=False,
    )
    ocr_result = document_ingest._extract_pdf_sections_sync(
        b"synthetic-pdf",
        ocr=True,
    )
    serialized = " ".join(parser_result.warnings + ocr_result.warnings)
    assert "PDF text extraction failed for page 1." in serialized
    assert "OCR processing failed or timed out for page 1." in serialized
    assert "test-secret" not in serialized
    assert "private.example" not in serialized
    assert synthetic_private_path.rsplit("/", 1)[0] not in serialized


def test_ocr_output_respects_incremental_character_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyPage:
        def extract_text(self) -> str:
            return ""

    class FakeReader:
        def __init__(self, *_args, **_kwargs) -> None:
            self.pages = [EmptyPage()]

    class FakeImage:
        def close(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakeReader))
    monkeypatch.setitem(
        sys.modules,
        "pdf2image",
        types.SimpleNamespace(
            convert_from_bytes=lambda *_args, **_kwargs: [FakeImage()],
            convert_from_path=lambda *_args, **_kwargs: [FakeImage()],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        types.SimpleNamespace(image_to_string=lambda *_args, **_kwargs: "too long"),
    )

    with pytest.raises(ValueError, match="OCR text exceeds"):
        document_ingest._extract_pdf_sections_sync(
            b"synthetic-pdf",
            ocr=True,
            max_extracted_chars=4,
        )
