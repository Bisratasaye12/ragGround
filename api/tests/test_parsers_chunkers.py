import io

import pytest

from ragground.ingestion.chunkers import FixedChunker, HeadingChunker
from ragground.ingestion.parsers import ParsedDocument, ParseError, Section, parse

MARKDOWN = """Intro paragraph before any heading.

# Install Guide

Top-level text.

## Linux

Run `apt install thing`.

```bash
# this is a comment inside code, not a heading
thing --start
```

### Debian

Debian specifics.

## macOS

Use brew.
"""


def test_markdown_keeps_heading_paths() -> None:
    doc = parse(MARKDOWN.encode(), "guide.md")
    assert doc.title == "Install Guide"
    paths = [s.heading_path for s in doc.sections]
    assert paths == [
        [],
        ["Install Guide"],
        ["Install Guide", "Linux"],
        ["Install Guide", "Linux", "Debian"],
        ["Install Guide", "macOS"],
    ]
    linux = doc.sections[2].text
    assert "# this is a comment inside code" in linux  # code fences aren't split as headings


def test_html_strips_boilerplate_and_keeps_headings() -> None:
    html = b"""<html><head><title>Payments API</title></head><body>
    <nav>Home | Pricing | Login</nav>
    <article><h1>Payments API</h1>
    <p>The payments API lets you create charges and refunds for customers in many currencies.</p>
    <h2>Refunds</h2>
    <p>Refunds are processed within five business days and returned to the original card.</p>
    </article><footer>Copyright 2026</footer></body></html>"""
    doc = parse(html, "payments.html")
    assert doc.title == "Payments API"
    text = doc.full_text
    assert "five business days" in text
    assert "Pricing" not in text
    assert ["Payments API", "Refunds"] in [s.heading_path for s in doc.sections]


def test_pdf_pages_become_sections() -> None:
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    for body in ("First page about invoices.", "Second page about refunds."):
        page = pdf.new_page()
        page.insert_text((72, 72), body)
    data = pdf.tobytes()
    doc = parse(data, "report.pdf")
    assert [s.heading_path for s in doc.sections] == [["Page 1"], ["Page 2"]]
    assert "refunds" in doc.sections[1].text


def test_docx_heading_styles() -> None:
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Leave Policy", level=1)
    d.add_paragraph("Employees get 25 days.")
    d.add_heading("Sick leave", level=2)
    d.add_paragraph("Sick leave is unlimited with a note.")
    buf = io.BytesIO()
    d.save(buf)
    doc = parse(buf.getvalue(), "policy.docx")
    assert doc.title == "Leave Policy"
    assert ["Leave Policy", "Sick leave"] in [s.heading_path for s in doc.sections]


@pytest.mark.parametrize(
    ("data", "name", "msg"),
    [
        (b"hello", "virus.exe", "unsupported"),
        (b"   \n ", "empty.txt", "empty"),
        (b"not a pdf at all", "broken.pdf", "PDF"),
    ],
)
def test_parse_errors(data: bytes, name: str, msg: str) -> None:
    with pytest.raises(ParseError, match=msg):
        parse(data, name)


def _doc(*sections: tuple[list[str], str]) -> ParsedDocument:
    return ParsedDocument("t", [Section(p, t) for p, t in sections])


def test_fixed_chunker_windows_and_overlap() -> None:
    words = " ".join(f"w{i}" for i in range(250))
    chunks = FixedChunker(size=100, overlap=20).chunk(_doc(([], words)))
    assert [c.word_count for c in chunks] == [100, 100, 90]
    assert chunks[1].text.split()[0] == "w80"  # 20-word overlap with chunk 0
    assert [c.ord for c in chunks] == [0, 1, 2]


def test_heading_chunker_respects_sections_and_limits() -> None:
    para = " ".join(["word"] * 60)
    doc = _doc(
        (["A"], "\n\n".join([para] * 3)),  # 180 words → must split at max 100
        (["B"], "short section with only a few words"),
    )
    chunks = HeadingChunker(max_len=100, min_len=10).chunk(doc)
    assert all(c.word_count <= 100 for c in chunks)
    assert {tuple(c.heading_path) for c in chunks} == {("A",), ("B",)}
    # no chunk mixes sections
    assert chunks[-1].heading_path == ["B"]
    assert chunks[-1].embedding_text.startswith("B\n\n")


def test_heading_chunker_splits_giant_paragraph_by_sentence() -> None:
    sentence = "This sentence has exactly eight words in it."
    para = " ".join([sentence] * 30)  # 240 words, one paragraph
    chunks = HeadingChunker(max_len=50, min_len=5).chunk(_doc((["S"], para)))
    assert all(c.word_count <= 50 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)  # split on sentence boundaries
    assert sum(c.word_count for c in chunks) == 240


def test_heading_chunker_merges_tiny_trailing_chunk() -> None:
    doc = _doc((["S"], "Tiny one.\n\nTiny two."))
    chunks = HeadingChunker(max_len=100, min_len=10).chunk(doc)
    assert len(chunks) == 1


def test_markdown_strips_inline_html_but_not_code() -> None:
    md = (
        b"# T\n\nInstall <a href='https://x.io' class=\"external-link\">`pkg`</a> first.\n\n"
        b"```html\n<div>keep me</div>\n```\n"
    )
    text = parse(md, "t.md").full_text
    assert "Install `pkg` first." in text
    assert "<a " not in text
    assert "<div>keep me</div>" in text  # code blocks untouched


def test_chunkers_respect_a_custom_length_function() -> None:
    """With a token-dense length (here: characters), limits hold for every chunk."""

    def chars(text: str) -> int:
        return len(text)

    code_heavy = " ".join(["response_model=list[schemas.Item]"] * 40)  # long "words"
    doc = _doc((["API"], f"Intro text.\n\n{code_heavy}\n\nOutro text here."))
    for chunker in (
        HeadingChunker(max_len=200, min_len=5, length=chars),
        FixedChunker(size=200, overlap=40, length=chars),
    ):
        chunks = chunker.chunk(doc)
        assert chunks
        # Per-word summing never under-counts, so the real length stays within the limit
        # (+ the single spaces joining words).
        assert all(len(c.text.replace(" ", "")) <= 200 for c in chunks), chunker
    fixed = FixedChunker(size=200, overlap=40, length=chars).chunk(doc)
    assert fixed[0].text.split()[-1] in fixed[1].text  # windows overlap
