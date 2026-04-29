import pytest

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
