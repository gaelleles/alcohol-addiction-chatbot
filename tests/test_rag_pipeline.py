from types import SimpleNamespace

import pytest

import rag_pipeline


def make_doc(**metadata):
    return SimpleNamespace(metadata=metadata)


# --- format_sources ---


def test_format_sources_converts_zero_indexed_page_to_one_indexed():
    docs = [make_doc(source="/data/paper.pdf", page=0)]
    assert rag_pipeline.format_sources(docs) == ["📄 paper.pdf (Page 1)"]


def test_format_sources_does_not_crash_when_page_is_missing():
    docs = [make_doc(source="/data/paper.pdf")]
    assert rag_pipeline.format_sources(docs) == ["📄 paper.pdf"]


def test_format_sources_deduplicates_and_sorts():
    docs = [
        make_doc(source="b.pdf", page=1),
        make_doc(source="a.pdf", page=4),
        make_doc(source="b.pdf", page=1),
    ]
    assert rag_pipeline.format_sources(docs) == [
        "📄 a.pdf (Page 5)",
        "📄 b.pdf (Page 2)",
    ]


# --- PRIVACY validation ---


@pytest.mark.parametrize(
    "raw, expected",
    [("public", "public"), ("PUBLIC", "public"), (" Private ", "private")],
)
def test_normalize_privacy_accepts_any_casing(raw, expected):
    assert rag_pipeline.normalize_privacy(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "prod"])
def test_normalize_privacy_rejects_missing_or_unknown_values(raw):
    with pytest.raises(ValueError, match="PRIVACY"):
        rag_pipeline.normalize_privacy(raw)


def test_get_pdf_directory_follows_privacy_setting():
    assert rag_pipeline.get_pdf_directory("PUBLIC") == "./pdf_papers/public/"
    assert rag_pipeline.get_pdf_directory("private") == "./pdf_papers/private/"


def test_persist_directory_depends_on_chunking_config():
    default = rag_pipeline.get_persist_directory("public")
    bigger = rag_pipeline.get_persist_directory("public", chunk_size=1000)
    assert default != bigger
    assert "public" in default


# --- cache invalidation ---


def test_fingerprint_changes_when_a_pdf_is_added_or_modified(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"one")
    before = rag_pipeline.compute_pdf_fingerprint(str(tmp_path))

    (tmp_path / "b.pdf").write_bytes(b"two")
    after_add = rag_pipeline.compute_pdf_fingerprint(str(tmp_path))
    assert after_add != before

    (tmp_path / "a.pdf").write_bytes(b"one, but longer")
    after_edit = rag_pipeline.compute_pdf_fingerprint(str(tmp_path))
    assert after_edit != after_add


def test_fingerprint_ignores_non_pdf_files(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"one")
    before = rag_pipeline.compute_pdf_fingerprint(str(tmp_path))
    (tmp_path / ".DS_Store").write_bytes(b"noise")
    assert rag_pipeline.compute_pdf_fingerprint(str(tmp_path)) == before


def test_cache_is_fresh_only_when_fingerprint_matches(tmp_path):
    assert not rag_pipeline.is_cache_fresh(str(tmp_path), "abc")

    (tmp_path / rag_pipeline.FINGERPRINT_FILENAME).write_text("abc")
    assert rag_pipeline.is_cache_fresh(str(tmp_path), "abc")
    assert not rag_pipeline.is_cache_fresh(str(tmp_path), "different")
