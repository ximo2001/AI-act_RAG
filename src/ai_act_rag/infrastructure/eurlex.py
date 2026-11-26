"""Download and parse the AI Act from EUR-Lex / the Official Journal XHTML.

The Official Journal (OJ) "CONVEX" XHTML, which EUR-Lex embeds in its HTML
view, marks every structural unit with stable ids:

* ``div#rct_N``                 recital N
* ``div#cpt_X`` / ``div#cpt_X.sct_Y``   chapter / section (titles in ``*.tit_1``)
* ``div#art_N``                 article N (``p.oj-ti-art`` + ``p.oj-sti-art`` title,
                                paragraphs in child ``div#NNN.MMM``)
* ``div#anx_X``                 annex X (``p.oj-doc-ti`` headings)

Relying on these ids (instead of regexes over plain text) makes the parser
robust and keeps article numbers, titles and chapters exact.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import tarfile
from collections.abc import Iterable
from pathlib import Path

import httpx
from lxml import etree, html

from ai_act_rag.domain.exceptions import ParsingError
from ai_act_rag.domain.models import LegalUnit, UnitType

logger = logging.getLogger(__name__)

CELEX = "32024R1689"
# Official sources, tried in order. The first is the EUR-Lex HTML view; the
# second asks the Publications Office Cellar for the XHTML manifestation.
EURLEX_URLS = (
    f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{CELEX}",
    f"https://publications.europa.eu/resource/celex/{CELEX}",
)
_CELLAR_HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}

_WS = re.compile(r"\s+")


def download_ai_act(dest: Path, urls: Iterable[str] = EURLEX_URLS, timeout: float = 60.0) -> Path:
    """Download the official AI Act text, trying each URL until one works.

    Args:
        dest: File to write (HTML/XHTML).
        urls: Candidate URLs, in order of preference.
        timeout: Per-request timeout in seconds.

    Returns:
        ``dest``, once written.

    Raises:
        RuntimeError: If every URL fails.
    """
    errors: list[str] = []
    for url in urls:
        try:
            logger.info("Downloading %s", url)
            resp = httpx.get(url, headers=_CELLAR_HEADERS, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            # Validate before writing: a cookie wall or error page must not be
            # silently accepted as the Regulation.
            units = parse_ai_act(resp.content)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            logger.info("Saved %s (%d units, sha256=%s)", dest, len(units), sha256(dest))
            return dest
        except (httpx.HTTPError, ParsingError) as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("Download failed: %s", errors[-1])
    raise RuntimeError("Could not download the AI Act:\n" + "\n".join(errors))


# Fallback mirror used when EUR-Lex is unreachable (e.g. behind a
# restricted network): the sdist of a PyPI package that vendors the Official
# Journal XHTML unchanged. Pinned by version AND by the SHA-256 of the file.
PYPI_MIRROR_URL = (
    "https://files.pythonhosted.org/packages/source/e/eu-ai-act-classifier/"
    "eu_ai_act_classifier-0.4.0.tar.gz"
)
PYPI_MIRROR_MEMBER = "eu_ai_act_classifier-0.4.0/sources/oj/32024R1689.en.xhtml"
PYPI_MIRROR_SHA256 = "8f0b656302f9864cc87e040c371f209a9d65ae1a6cecc25ca5eb737e872d721a"


def download_from_pypi_mirror(dest: Path, timeout: float = 120.0) -> Path:
    """Fetch the OJ XHTML vendored in a PyPI sdist and verify its hash.

    Args:
        dest: File to write.
        timeout: HTTP timeout in seconds.

    Returns:
        ``dest``, once written and verified.

    Raises:
        RuntimeError: If the download fails or the hash does not match.
    """
    resp = httpx.get(PYPI_MIRROR_URL, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        member = tar.extractfile(PYPI_MIRROR_MEMBER)
        if member is None:
            raise RuntimeError(f"{PYPI_MIRROR_MEMBER} not found in the sdist")
        content = member.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != PYPI_MIRROR_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: {digest} != {PYPI_MIRROR_SHA256}")
    parse_ai_act(content)  # structural validation
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


def sha256(path: Path) -> str:
    """Compute the SHA-256 of a file.

    Args:
        path: File path.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(text: str) -> str:
    """Normalise whitespace and drop stray characters present in the OJ text.

    Args:
        text: Raw text.

    Returns:
        Single-spaced, stripped text.
    """
    # The OJ XHTML contains non-breaking spaces (U+00A0); written as an escape
    # because the literal character is invisible in the source code.
    return _WS.sub(" ", text.replace("\u00a0", " ")).strip()


def _text(el: etree._Element) -> str:
    """Return the normalised text content of an element.

    Args:
        el: Any element.

    Returns:
        Its whitespace-normalised text.
    """
    return _clean("".join(str(t) for t in el.itertext()))


def _render(el: etree._Element) -> list[str]:
    """Render an element into text lines, keeping list labels such as ``(a)``.

    OJ lists are two-column tables (label | content). We emit
    ``"(a) content"`` so points stay attached to their labels; nested lists
    are rendered recursively.

    Args:
        el: Element to render.

    Returns:
        Lines of text in document order.
    """
    tag = el.tag if isinstance(el.tag, str) else ""
    tag = tag.split("}")[-1]
    if tag == "p":
        txt = _text(el)
        return [txt] if txt else []
    if tag == "table":
        lines: list[str] = []
        for row in el.iter("{*}tr", "tr"):
            # Only direct rows of this table (nested tables are handled by recursion).
            if _closest_table(row) is not el:
                continue
            cells = [c for c in row if isinstance(c.tag, str) and c.tag.split("}")[-1] == "td"]
            if len(cells) >= 2:
                label = _text(cells[0])
                body: list[str] = []
                for child in cells[1]:
                    body.extend(_render(child))
                if body:
                    lines.append(f"{label} {body[0]}".strip())
                    lines.extend(body[1:])
            elif cells:
                for child in cells[0]:
                    lines.extend(_render(child))
        return lines
    out: list[str] = []
    for child in el:
        out.extend(_render(child))
    return out


def _closest_table(el: etree._Element) -> etree._Element | None:
    """Return the nearest ``table`` ancestor of an element.

    Args:
        el: Element (typically a ``tr``).

    Returns:
        The enclosing table, or None.
    """
    parent = el.getparent()
    while parent is not None:
        if isinstance(parent.tag, str) and parent.tag.split("}")[-1] == "table":
            return parent
        parent = parent.getparent()
    return None


def _find_by_class(root: etree._Element, css_class: str) -> list[etree._Element]:
    """Return all elements whose ``class`` attribute contains ``css_class`` as a word.

    Args:
        root: Document root.
        css_class: Class name, e.g. ``"oj-super"``.

    Returns:
        Matching elements in document order.
    """
    return [el for el in root.iter() if css_class in (el.get("class") or "").split()]


def _strip_footnote_refs(root: etree._Element) -> None:
    """Remove footnote markers like ``(4)`` that point to OJ footnotes.

    Args:
        root: Document root (modified in place).
    """
    for span in _find_by_class(root, "oj-note-tag"):
        anchor = span.getparent()
        is_anchor = anchor is not None and str(anchor.tag).split("}")[-1] == "a"
        target = anchor if anchor is not None and is_anchor else span
        parent = target.getparent()
        if parent is None:
            continue
        # Preserve the tail text (the words that follow the marker).
        tail = target.tail or ""
        prev = target.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
        parent.remove(target)


def _mark_superscripts(root: etree._Element) -> None:
    """Render exponent superscripts as ``^``: ``10<sup>25</sup>`` -> ``10^25``.

    Without this, "10^25 FLOPs" in Article 51 would read "1025".

    Args:
        root: Document root (modified in place; call after footnotes are removed).
    """
    for span in _find_by_class(root, "oj-super"):
        span.text = "^" + (span.text or "")


def _by_id(root: etree._Element, prefix: str) -> list[etree._Element]:
    """Find elements whose id is ``prefix`` followed only by a number/roman numeral.

    Args:
        root: Document root.
        prefix: e.g. ``"art_"``.

    Returns:
        Matching elements in document order.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}[0-9IVXLC]+$")
    return [el for el in root.iter() if pattern.match(el.get("id") or "")]


def _title_of(container: etree._Element) -> str | None:
    """Return the title text of a chapter/section container (its ``*.tit_1`` child).

    Args:
        container: ``div#cpt_X`` or ``div#cpt_X.sct_Y``.

    Returns:
        The title, or None.
    """
    cid = container.get("id")
    for child in container:
        if child.get("id") == f"{cid}.tit_1":
            return _sentence_case(_text(child))
    return None


_KEEP_UPPER = {"AI", "EU", "SMES", "SMEs"}


def _sentence_case(title: str) -> str:
    """Turn an all-caps OJ heading into sentence case, keeping acronyms.

    Args:
        title: e.g. ``"PROHIBITED AI PRACTICES"``.

    Returns:
        e.g. ``"Prohibited AI practices"``; mixed-case input is returned as is.
    """
    if not title.isupper():
        return title
    words = [w if w in _KEEP_UPPER else w.lower() for w in title.split()]
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def _heading_of(container: etree._Element) -> str:
    """Return the "CHAPTER III" / "SECTION 1" heading of a container.

    Args:
        container: Chapter or section div.

    Returns:
        Heading text in title case ("Chapter III").
    """
    for child in container:
        if "oj-ti-section-1" in (child.get("class") or ""):
            kind, _, number = _text(child).partition(" ")
            return f"{kind.capitalize()} {number}"
    return container.get("id") or ""


def _parse_recitals(root: etree._Element) -> list[LegalUnit]:
    """Extract recitals ``(1)``..``(180)``.

    Args:
        root: Document root.

    Returns:
        One unit per recital.
    """
    units = []
    for div in _by_id(root, "rct_"):
        number = div.get("id", "").removeprefix("rct_")
        lines = _render(div)
        # The rendered line starts with the "(N)" label; drop it.
        text = re.sub(rf"^\({number}\)\s*", "", " ".join(lines))
        units.append(
            LegalUnit(
                unit_id=f"Recital {number}",
                unit_type=UnitType.RECITAL,
                number=number,
                paragraphs=[text],
            )
        )
    return units


def _parse_articles(root: etree._Element) -> list[LegalUnit]:
    """Extract articles with title, chapter, section and paragraphs.

    Args:
        root: Document root.

    Returns:
        One unit per article.
    """
    units = []
    for div in _by_id(root, "art_"):
        number = div.get("id", "").removeprefix("art_")
        title_el = div.find(f".//*[@id='art_{number}.tit_1']")
        title = _text(title_el).strip("`' ") if title_el is not None else None
        chapter = section = None
        for anc in div.iterancestors():
            aid = anc.get("id") or ""
            if re.fullmatch(r"cpt_[IVXLC]+\.sct_\d+", aid) and section is None:
                section = f"{_heading_of(anc)} - {_title_of(anc)}"
            elif re.fullmatch(r"cpt_[IVXLC]+", aid):
                chapter = f"{_heading_of(anc)} - {_title_of(anc)}"
                break
        paragraphs: list[str] = []
        for child in div:
            cls = child.get("class") or ""
            if "oj-ti-art" in cls or child.get("id") == f"art_{number}.tit_1":
                continue
            lines = _render(child)
            if lines:
                paragraphs.append("\n".join(lines))
        if not paragraphs:
            raise ParsingError(f"Article {number} has no text")
        units.append(
            LegalUnit(
                unit_id=f"Article {number}",
                unit_type=UnitType.ARTICLE,
                number=number,
                title=title,
                chapter=chapter,
                section=section,
                paragraphs=paragraphs,
            )
        )
    return units


def _parse_annexes(root: etree._Element) -> list[LegalUnit]:
    """Extract annexes; each top-level block (point, table, heading) is a paragraph.

    Args:
        root: Document root.

    Returns:
        One unit per annex.
    """
    units = []
    for div in _by_id(root, "anx_"):
        number = div.get("id", "").removeprefix("anx_")
        headings = [c for c in div if "oj-doc-ti" in (c.get("class") or "")]
        title = _text(headings[1]) if len(headings) > 1 else None
        paragraphs: list[str] = []
        pending_heading: str | None = None
        for child in div:
            if child in headings:
                continue
            lines = _render(child)
            if not lines:
                continue
            block = "\n".join(lines)
            # Short sub-headings ("Section A", "PART 1") are glued to the next block.
            if "oj-ti-grseq-1" in (child.get("class") or "") or (
                len(lines) == 1 and len(block) < 80 and not block.endswith((".", ";", ":"))
            ):
                pending_heading = f"{pending_heading}\n{block}" if pending_heading else block
                continue
            if pending_heading:
                block = f"{pending_heading}\n{block}"
                pending_heading = None
            paragraphs.append(block)
        if pending_heading:
            paragraphs.append(pending_heading)
        units.append(
            LegalUnit(
                unit_id=f"Annex {number}",
                unit_type=UnitType.ANNEX,
                number=number,
                title=title,
                paragraphs=paragraphs,
            )
        )
    return units


EXPECTED_COUNTS = (113, 180, 13)  # articles, recitals, annexes of Regulation 2024/1689


def parse_ai_act(
    content: bytes, expected_counts: tuple[int, int, int] | None = EXPECTED_COUNTS
) -> list[LegalUnit]:
    """Parse the AI Act (OJ XHTML or EUR-Lex HTML) into legal units.

    Args:
        content: Raw HTML/XHTML bytes.
        expected_counts: (articles, recitals, annexes) to validate against, or
            None to skip the check (used by tests with small fixtures).

    Returns:
        Recitals, then articles, then annexes, in document order.

    Raises:
        ParsingError: If the document does not look like Regulation 2024/1689.
    """
    try:
        root = html.fromstring(content)
    except (etree.ParserError, ValueError) as exc:
        raise ParsingError(f"Not an HTML document: {exc}") from exc
    _strip_footnote_refs(root)
    _mark_superscripts(root)
    recitals = _parse_recitals(root)
    articles = _parse_articles(root)
    annexes = _parse_annexes(root)
    # Sanity checks against the published structure of Regulation 2024/1689.
    found = (len(articles), len(recitals), len(annexes))
    if expected_counts is not None and found != expected_counts:
        raise ParsingError(
            f"Unexpected structure (articles, recitals, annexes) = {found}, "
            f"expected {expected_counts}"
        )
    return [*recitals, *articles, *annexes]
