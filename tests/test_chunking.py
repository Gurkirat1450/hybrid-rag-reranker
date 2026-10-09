"""Tests for app/chunking.py."""

from pathlib import Path

import pytest

from app.chunking import (
    Chunk,
    chunk_corpus,
    chunk_markdown,
    clean_heading,
    embedding_text,
    overlap_tail,
    split_long,
    strip_noise,
)

SAMPLE = "\n".join(
    [
        "# My Doc { #my-doc }",
        "",
        "Intro paragraph.",
        "",
        "{* ../../docs_src/x.py hl[1] *}",
        "",
        "/// tip",
        "",
        "Tip text.",
        "",
        "///",
        "",
        "## Section A { #section-a }",
        "",
        "Body A.",
        "",
        "```python",
        "## not a heading",
        "```",
        "",
        "### Short { #short }",
        "",
        "tiny",
    ]
)


def test_strip_noise_drops_mkdocs_lines_only():
    out = strip_noise("keep\n{* include *}\n/// tip\nalso keep\n///\n{! snippet !}\n  {* indented *}")
    assert out == "keep\nalso keep"


def test_clean_heading_removes_anchor():
    assert clean_heading("Query Parameters { #query-parameters }") == "Query Parameters"
    assert clean_heading("No anchor") == "No anchor"


def test_sections_headings_and_code_fences():
    # min_chars=0 disables merging, so we see every section separately.
    chunks = chunk_markdown(SAMPLE, "my-doc.md", min_chars=0)
    assert chunks == [
        # noise lines removed; the blank lines left behind collapse to one blank line
        ("My Doc", "Intro paragraph.\n\nTip text."),
        # "## not a heading" is inside a code fence, so it does not start a section
        ("My Doc > Section A", "Body A.\n\n```python\n## not a heading\n```"),
        ("My Doc > Short", "tiny"),
    ]


def test_small_sections_merge_into_previous_chunk():
    # default min_chars=200: every body here is shorter, so Section A and Short merge
    # into the intro chunk (which has no previous chunk, so it stays on its own).
    chunks = chunk_markdown(SAMPLE, "my-doc.md")
    assert chunks == [
        (
            "My Doc",
            "Intro paragraph.\n\nTip text."
            "\n\nSection A\nBody A.\n\n```python\n## not a heading\n```"
            "\n\nShort\ntiny",
        )
    ]


def test_title_falls_back_to_file_stem():
    assert chunk_markdown("Just text here.", "plain.md", min_chars=0) == [("plain", "Just text here.")]


def test_level4_headings_stay_in_section_and_empty_intro_is_skipped():
    md = "# T\n\n## S\n\nx\n\n#### Deep\n\ny"
    assert chunk_markdown(md, "t.md", min_chars=0) == [("T > S", "x\n\n#### Deep\n\ny")]


def test_split_long_short_text_is_unchanged():
    assert split_long("short", max_chars=10, overlap=3) == ["short"]


def test_split_long_on_paragraphs_without_overlap():
    # "aaaa\n\nbbbb\n\ncccc" is 4+2+4+2+4 = 16 chars > 10
    # pack: "aaaa" (4) + "\n\nbbbb" (6) = 10 <= 10; adding "\n\ncccc" would be 16 > 10
    assert split_long("aaaa\n\nbbbb\n\ncccc", max_chars=10, overlap=0) == ["aaaa\n\nbbbb", "cccc"]


def test_split_long_with_overlap():
    # paragraphs: "alpha beta" (10), "gamma delta" (11), "epsilon zeta" (12); total 10+2+11+2+12 = 37
    # max 25: 10 + 2 + 11 = 23 fits; + 2 + 12 = 37 does not -> piece 1 = "alpha beta\n\ngamma delta"
    # overlap 5: last 5 chars of piece 1 = "delta" (preceded by a space, so no partial word)
    # piece 2 = "delta" + "\n\n" + "epsilon zeta" = 19 chars
    text = "alpha beta\n\ngamma delta\n\nepsilon zeta"
    pieces = split_long(text, max_chars=25, overlap=5)
    assert pieces == ["alpha beta\n\ngamma delta", "delta\n\nepsilon zeta"]
    assert all(len(p) <= 25 for p in pieces)


def test_split_long_hard_splits_oversized_paragraph():
    # one 14-char paragraph, max 10, overlap 0 -> unit limit 10 - 0 - 2 = 8
    # hard split at spaces: "aaaa", "bbbb", "cccc"; then pack as above
    assert split_long("aaaa bbbb cccc", max_chars=10, overlap=0) == ["aaaa\n\nbbbb", "cccc"]


def test_overlap_tail_word_boundaries():
    assert overlap_tail("hello world", 5) == "world"  # char before is a space: clean cut
    assert overlap_tail("hello world", 7) == "world"  # "o world" starts mid-word -> drop "o"
    assert overlap_tail("hello world", 4) == ""  # "orld" is a partial word only
    assert overlap_tail("hi", 10) == "hi"
    assert overlap_tail("abc", 0) == ""


def test_embedding_text():
    assert embedding_text(Chunk(0, "a.md", "A > B", "body")) == "A > B\nbody"


def test_chunk_corpus_ids_are_deterministic_and_file_sorted(tmp_path: Path):
    (tmp_path / "b.md").write_text("# B\n\nbee text", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\nalpha text\n\n## Two\n\nsecond", encoding="utf-8")
    first = chunk_corpus(tmp_path, min_chars=0)
    second = chunk_corpus(tmp_path, min_chars=0)
    assert first == second
    assert [(c.chunk_id, c.doc_id, c.heading, c.text) for c in first] == [
        (0, "a.md", "A", "alpha text"),
        (1, "a.md", "A > Two", "second"),
        (2, "b.md", "B", "bee text"),
    ]


CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@pytest.mark.skipif(not CORPUS.exists(), reason="data/corpus not present")
def test_real_corpus_is_clean():
    chunks = chunk_corpus(CORPUS)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert {c.doc_id for c in chunks} == {p.name for p in CORPUS.glob("*.md")}
    for c in chunks:
        assert "{ #" not in c.heading
        assert c.text.strip()
        for line in c.text.splitlines():
            assert not line.lstrip().startswith(("{*", "///", "{!"))
