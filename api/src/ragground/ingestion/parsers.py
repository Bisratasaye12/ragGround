"""Turn raw file bytes into a structured `ParsedDocument`: a title plus ordered sections, each
carrying its heading path (e.g. ["Install", "Linux"]). Structure is kept so the heading-aware
chunker can split on section boundaries and prefix chunks with their headings.
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath


class ParseError(Exception):
    """The file can't be parsed. Permanent: retrying won't help."""


@dataclass
class Section:
    heading_path: list[str]
    text: str


@dataclass
class ParsedDocument:
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm", ".docx"}


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_FENCE = re.compile(r"(^```.*?^```[^\n]*$)", re.S | re.M)


def strip_inline_html(markdown: str) -> str:
    """Drop inline HTML tags (keeping their text) outside fenced code blocks. Docs often embed
    `<a href=…>` / `<abbr>` markup that is pure noise for embeddings and prompts."""
    parts = _FENCE.split(markdown)
    return "".join(p if p.startswith("```") else _HTML_TAG.sub("", p) for p in parts)


def parse_markdown(text: str, fallback_title: str) -> ParsedDocument:
    from markdown_it import MarkdownIt

    text = strip_inline_html(text)

    md = MarkdownIt("commonmark")
    lines = text.splitlines()
    tokens = md.parse(text)
    # (line where heading starts, line after heading, level, heading text)
    headings: list[tuple[int, int, int, str]] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.map:
            inline = tokens[i + 1].content.strip()
            headings.append((tok.map[0], tok.map[1], int(tok.tag[1]), inline))

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    def add(path: list[str], start: int, end: int) -> None:
        body = _clean("\n".join(lines[start:end]))
        if body:
            sections.append(Section(heading_path=path, text=body))

    first_start = headings[0][0] if headings else len(lines)
    add([], 0, first_start)  # preamble before the first heading
    for idx, (_, body_start, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        add([t for _, t in stack], body_start, end)

    h1 = next((t for _, _, lvl, t in headings if lvl == 1), None)
    return ParsedDocument(title=h1 or fallback_title, sections=sections)


def parse_pdf(data: bytes, fallback_title: str) -> ParsedDocument:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ParseError(f"not a readable PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise ParseError("PDF is password-protected")
        title = (doc.metadata or {}).get("title") or fallback_title
        sections = [
            Section(heading_path=[f"Page {n + 1}"], text=text)
            for n in range(doc.page_count)
            if (text := _clean(str(doc[n].get_text())))
        ]
    if not sections:
        raise ParseError("PDF has no extractable text (scanned? OCR is not supported yet)")
    return ParsedDocument(title=title, sections=sections)


def parse_html(text: str, fallback_title: str) -> ParsedDocument:
    import trafilatura

    # Strips nav/boilerplate and keeps headings as Markdown, so it reuses the Markdown parser.
    extracted = trafilatura.extract(
        text, output_format="markdown", include_links=False, include_tables=True
    )
    if not extracted:
        raise ParseError("no main content found in HTML")
    meta = trafilatura.extract_metadata(text)
    parsed = parse_markdown(extracted, fallback_title)
    if meta is not None and meta.title:
        parsed.title = meta.title
    return parsed


def parse_docx(data: bytes, fallback_title: str) -> ParsedDocument:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ParseError(f"not a readable DOCX: {exc}") from exc
    # Rebuild as Markdown from paragraph styles ("Heading 1".. / "Title"), then reuse that parser.
    out: list[str] = []
    for para in document.paragraphs:
        style = (para.style.name if para.style is not None else "") or ""
        text = para.text.strip()
        if not text:
            continue
        if style == "Title":
            out.append(f"# {text}")
        elif m := re.match(r"Heading (\d)", style):
            out.append(f"{'#' * min(int(m.group(1)), 6)} {text}")
        else:
            out.append(text)
        out.append("")
    title = document.core_properties.title or fallback_title
    parsed = parse_markdown("\n".join(out), title)
    if not parsed.sections:
        raise ParseError("DOCX has no text")
    return parsed


def parse(data: bytes, filename: str) -> ParsedDocument:
    ext = PurePosixPath(filename).suffix.lower()
    stem = PurePosixPath(filename).stem or filename
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"unsupported file type {ext or '(none)'}")
    if ext == ".pdf":
        parsed = parse_pdf(data, stem)
    elif ext == ".docx":
        parsed = parse_docx(data, stem)
    elif ext in {".html", ".htm"}:
        parsed = parse_html(_decode(data), stem)
    elif ext in {".md", ".markdown"}:
        parsed = parse_markdown(_decode(data), stem)
    else:
        text = _clean(_decode(data))
        parsed = ParsedDocument(title=stem, sections=[Section([], text)] if text else [])
    if not parsed.sections:
        raise ParseError("document is empty")
    return parsed
