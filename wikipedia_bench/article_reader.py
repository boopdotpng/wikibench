"""On-the-fly article reader from Wikipedia multistream XML dump.

Reads individual articles by seeking to the right bz2 block using the
multistream index. Caches recently-read blocks and articles via LRU.
Returns section-structured text with inline [[links]].
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .text_extract import (
    build_offset_index,
    extract_pages_from_block,
    read_bz2_block,
)

_INDEX_CACHE_MAGIC_V1 = b'WBIDX001'
_INDEX_CACHE_MAGIC_V2 = b'WBIDX002'
_INDEX_CACHE_ENTRY = struct.Struct('<IQ')  # page_id (uint32), offset (uint64)
_INDEX_V2_DTYPE = np.dtype([('pid', '<u4'), ('offset', '<u8')])
_INDEX_V2_HEADER_SIZE = 8 + 4  # magic (8) + count (4) = 12 bytes

# Regexes for wikitext cleanup (keep links intact)
_RE_REF = re.compile(r'<ref[^>]*/>', re.DOTALL)
_RE_REF_BLOCK = re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL)
_RE_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)
_RE_GALLERY = re.compile(r'<gallery[^>]*>.*?</gallery>', re.DOTALL)
_RE_MATH = re.compile(r'<math[^>]*>.*?</math>', re.DOTALL)
_RE_NOWIKI = re.compile(r'<nowiki[^>]*>.*?</nowiki>', re.DOTALL)
_RE_TEMPLATE = re.compile(r'\{\{[^{}]*\}\}')
_RE_FILE_LINK = re.compile(r'\[\[(?:File|Image):[^\]]*\]\]', re.IGNORECASE)
_RE_CATEGORY = re.compile(r'\[\[Category:[^\]]*\]\]', re.IGNORECASE)
_RE_HTML_TAG = re.compile(r'<[^>]+>')
_RE_MULTI_NEWLINE = re.compile(r'\n{3,}')
_RE_MULTI_SPACE = re.compile(r'  +')
_RE_HEADING = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.MULTILINE)

# Navbox prefixes to strip from end
_NAVBOX_PREFIXES = (
    'navbox', 'authority control', 'taxonbar', 'portal bar',
    'portal', 'commons category', 'wikiquote', 'wikisource',
    'wiktionary', 'reflist', 'notelist', 'refbegin', 'refend',
    'defaultsort', 'short description', 'use dmy', 'use mdy',
)


class ArticleReader:
    """Reads full article text on-the-fly from the multistream dump."""

    def __init__(self, dump_path: Path, index_path: Path):
        self._dump_path = dump_path
        self._index_path = index_path
        # Lazily loaded: sorted numpy structured array (pid, offset)
        self._index_pids: np.ndarray | None = None     # sorted uint32 page_ids
        self._index_offsets: np.ndarray | None = None   # corresponding uint64 offsets
        # LRU caches
        self._read_block = lru_cache(maxsize=32)(self._read_block_uncached)
        self._article_cache: dict[int, str | None] = {}

    def load_index(self) -> None:
        """Load the multistream index. Uses mmap'd v2 binary cache for speed."""
        if self._index_pids is not None:
            return

        cache_path = self._index_path.with_suffix('.cache')
        index_mtime = self._index_path.stat().st_mtime

        # Try loading from v2 binary cache (mmap, sorted)
        if cache_path.exists() and cache_path.stat().st_mtime >= index_mtime:
            try:
                pids, offsets = _load_index_cache_v2(cache_path)
                self._index_pids = pids
                self._index_offsets = offsets
                return
            except Exception:
                pass  # Fall through to rebuild

        # Parse from bz2 and save v2 cache
        pid_to_offset = build_offset_index(self._index_path)
        try:
            _save_index_cache_v2(pid_to_offset, cache_path)
        except Exception:
            pass  # Non-fatal

        # Build sorted arrays from the dict
        pids = np.array(sorted(pid_to_offset.keys()), dtype=np.uint32)
        offsets = np.array([pid_to_offset[p] for p in pids], dtype=np.uint64)
        self._index_pids = pids
        self._index_offsets = offsets

    def _ensure_index(self) -> None:
        """Load the multistream index (lazy fallback)."""
        if self._index_pids is not None:
            return
        self.load_index()

    def _lookup_offset(self, page_id: int) -> int | None:
        """Binary search for a page_id's byte offset. Returns None if not found."""
        idx = np.searchsorted(self._index_pids, page_id)
        if idx < len(self._index_pids) and self._index_pids[idx] == page_id:
            return int(self._index_offsets[idx])
        return None

    def _read_block_uncached(self, offset: int) -> bytes:
        return read_bz2_block(self._dump_path, offset)

    def get_raw_wikitext(self, page_id: int) -> str | None:
        """Get raw wikitext for a page_id."""
        if page_id in self._article_cache:
            return self._article_cache[page_id]

        self._ensure_index()
        offset = self._lookup_offset(page_id)
        if offset is None:
            self._article_cache[page_id] = None
            return None

        block_data = self._read_block(offset)
        pages = extract_pages_from_block(block_data, {page_id})
        text = pages.get(page_id)
        self._article_cache[page_id] = text

        # Also cache other pages from the same block (they're free)
        for pid, wikitext in pages.items():
            if pid not in self._article_cache:
                self._article_cache[pid] = wikitext

        return text

    def get_article_text(self, page_id: int) -> str | None:
        """Get cleaned, section-structured article text with inline links.

        Returns text like:
            Article text with [[Link_Target|display text]] preserved...

            == History ==
            History text with [[World_War_II]] links...

            == Geography ==
            ...
        """
        raw = self.get_raw_wikitext(page_id)
        if raw is None:
            return None
        return _process_wikitext(raw)


def _process_wikitext(text: str) -> str:
    """Clean wikitext, preserving section structure and inline links."""
    # Extract infobox before stripping templates
    infobox_text = _extract_infobox(text)

    # Strip navbox/footer region
    text = _strip_navbox_region(text)

    # Strip HTML-like tags we don't want
    text = _RE_REF.sub('', text)
    text = _RE_REF_BLOCK.sub('', text)
    text = _RE_COMMENT.sub('', text)
    text = _RE_GALLERY.sub('', text)
    text = _RE_MATH.sub('', text)
    text = _RE_NOWIKI.sub('', text)

    # Strip File/Image and Category links
    text = _RE_FILE_LINK.sub('', text)
    text = _RE_CATEGORY.sub('', text)

    # Strip templates iteratively
    for _ in range(10):
        new = _RE_TEMPLATE.sub('', text)
        if new == text:
            break
        text = new

    # Strip bold/italic markup
    text = text.replace("'''", '').replace("''", '')

    # Strip remaining HTML tags
    text = _RE_HTML_TAG.sub('', text)

    # Clean up whitespace
    text = _RE_MULTI_SPACE.sub(' ', text)
    text = _RE_MULTI_NEWLINE.sub('\n\n', text)

    # Clean up empty lines and leading spaces per line
    lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        elif lines and lines[-1] != '':
            lines.append('')
    text = '\n'.join(lines).strip()

    # Prepend infobox if we found one
    if infobox_text:
        text = infobox_text + '\n\n' + text

    return text


# Keys to skip in infobox rendering (images, layout, internal metadata)
_INFOBOX_SKIP_KEYS = {
    'image', 'image_size', 'imagesize', 'image_caption', 'caption',
    'image_width', 'image_alt', 'alt', 'logo', 'logo_size', 'logo_alt',
    'photo', 'photo_size', 'photo_caption', 'embed', 'background',
    'module', 'width', 'header', 'above', 'below', 'child',
    'bodystyle', 'labelstyle', 'datastyle', 'headerstyle',
    'abovestyle', 'belowstyle', 'titlestyle',
}


def _extract_infobox(text: str) -> str:
    """Extract and render the first Infobox template as key-value pairs.

    Returns a string like:
        [Infobox: person]
        name: Erika Heynatz
        birth_date: 25 March 1975
        birth_place: [[Port Moresby]], [[Territory of Papua New Guinea]]
        nationality: Australian
        occupation: Singer, actress, model, television personality
    """
    # Find the start of an Infobox template
    match = re.search(r'\{\{[Ii]nfobox\s+([^|{}]+)', text)
    if not match:
        return ''

    infobox_type = match.group(1).strip()
    start = match.start()

    # Find the matching closing }} by counting brace depth
    depth = 0
    i = start
    end = len(text)
    while i < end:
        if text[i:i+2] == '{{':
            depth += 1
            i += 2
        elif text[i:i+2] == '}}':
            depth -= 1
            if depth == 0:
                i += 2
                break
            i += 2
        else:
            i += 1

    infobox_raw = text[start:i]

    # Parse key = value pairs from the infobox
    # Split on | that are at depth 0 (not inside nested {{...}})
    pairs = _split_infobox_fields(infobox_raw)

    lines = [f'[Infobox: {infobox_type}]']
    for key, value in pairs:
        key_lower = key.lower().strip()
        if key_lower in _INFOBOX_SKIP_KEYS:
            continue
        if not value.strip():
            continue

        # Clean the value: strip nested templates, refs, but keep [[links]]
        val = _expand_date_templates(value)
        val = _RE_REF.sub('', val)
        val = _RE_REF_BLOCK.sub('', val)
        val = _RE_COMMENT.sub('', val)
        val = _RE_HTML_TAG.sub('', val)
        # Strip templates inside values
        for _ in range(5):
            new = _RE_TEMPLATE.sub('', val)
            if new == val:
                break
            val = new
        # Clean Flatlist markers
        val = val.replace('*', ', ').replace('\n', ' ')
        val = _RE_MULTI_SPACE.sub(' ', val).strip()
        val = val.strip(',').strip()
        if val:
            lines.append(f'  {key.strip()}: {val}')

    if len(lines) <= 1:
        return ''

    return '\n'.join(lines)


_MONTHS = {
    '1': 'January', '2': 'February', '3': 'March', '4': 'April',
    '5': 'May', '6': 'June', '7': 'July', '8': 'August',
    '9': 'September', '10': 'October', '11': 'November', '12': 'December',
}

_RE_BIRTH_DATE = re.compile(
    r'\{\{(?:birth date and age|birth date|birth-date and age|bda)'
    r'[^}]*\|(\d{4})\|(\d{1,2})\|(\d{1,2})[^}]*\}\}',
    re.IGNORECASE,
)
_RE_DEATH_DATE = re.compile(
    r'\{\{(?:death date and age|death date|death-date and age|dda)'
    r'[^}]*\|(\d{4})\|(\d{1,2})\|(\d{1,2})[^}]*\}\}',
    re.IGNORECASE,
)
_RE_START_DATE = re.compile(
    r'\{\{(?:start date)[^}]*\|(\d{4})\|(\d{1,2})\|(\d{1,2})[^}]*\}\}',
    re.IGNORECASE,
)


def _expand_date_templates(text: str) -> str:
    """Expand common date templates to human-readable text."""
    def _fmt_date(m: re.Match) -> str:
        year, month, day = m.group(1), m.group(2), m.group(3)
        month_name = _MONTHS.get(month, month)
        return f'{int(day)} {month_name} {year}'

    text = _RE_BIRTH_DATE.sub(_fmt_date, text)
    text = _RE_DEATH_DATE.sub(_fmt_date, text)
    text = _RE_START_DATE.sub(_fmt_date, text)
    return text


def _split_infobox_fields(infobox: str) -> list[tuple[str, str]]:
    """Split an infobox template into (key, value) pairs.

    Handles nested {{ }} by tracking brace depth.
    """
    # Remove the outer {{ Infobox ... and closing }}
    # Find first | after the infobox name
    first_pipe = -1
    depth = 0
    for i, ch in enumerate(infobox):
        if infobox[i:i+2] == '{{':
            depth += 1
        elif infobox[i:i+2] == '}}':
            depth -= 1
        elif ch == '|' and depth == 1:
            first_pipe = i
            break

    if first_pipe == -1:
        return []

    body = infobox[first_pipe + 1:]
    # Remove trailing }}
    if body.rstrip().endswith('}}'):
        body = body.rstrip()[:-2]

    # Split on | at depth 0
    fields = []
    current = []
    depth = 0
    for ch in body:
        if ch == '{' and len(current) > 0 and current[-1] == '{':
            depth += 1
            current.append(ch)
        elif ch == '}' and len(current) > 0 and current[-1] == '}':
            depth -= 1
            current.append(ch)
        elif ch == '|' and depth <= 0:
            fields.append(''.join(current))
            current = []
            depth = 0  # reset in case of mismatch
        else:
            current.append(ch)

    if current:
        fields.append(''.join(current))

    # Parse key = value
    pairs = []
    for field in fields:
        if '=' in field:
            key, _, value = field.partition('=')
            pairs.append((key.strip(), value.strip()))

    return pairs


def _strip_navbox_region(wikitext: str) -> str:
    """Remove navbox/footer template regions from the end of wikitext."""
    lines = wikitext.split('\n')
    cutoff = len(lines)

    i = len(lines) - 1
    while i >= 0:
        stripped = lines[i].strip().lower()
        if stripped.startswith('{{'):
            template_name = stripped[2:].split('|')[0].split('}')[0].strip()
            if any(template_name.startswith(p) for p in _NAVBOX_PREFIXES):
                cutoff = i
                i -= 1
                continue
        if stripped == '' or stripped.startswith('[[category:'):
            cutoff = i
            i -= 1
            continue
        break

    return '\n'.join(lines[:cutoff])


def _save_index_cache_v2(pid_to_offset: dict[int, int], cache_path: Path) -> None:
    """Save pid->offset mapping as a sorted, mmap-friendly binary file.

    Format v2: 8-byte magic 'WBIDX002' + uint32 count + sorted array of
    (uint32 pid, uint64 offset) records. Sorted by pid for binary search.
    """
    count = len(pid_to_offset)
    arr = np.empty(count, dtype=_INDEX_V2_DTYPE)
    for i, (pid, offset) in enumerate(pid_to_offset.items()):
        arr[i] = (pid, offset)
    arr.sort(order='pid')

    with open(cache_path, 'wb') as f:
        f.write(_INDEX_CACHE_MAGIC_V2)
        f.write(struct.pack('<I', count))
        arr.tofile(f)


def _load_index_cache_v2(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pid->offset mapping via mmap. Returns (pids, offsets) arrays."""
    with open(cache_path, 'rb') as f:
        magic = f.read(8)
        if magic == _INDEX_CACHE_MAGIC_V1:
            raise ValueError('v1 cache — needs rebuild')
        if magic != _INDEX_CACHE_MAGIC_V2:
            raise ValueError('bad magic')

    arr = np.memmap(cache_path, dtype=_INDEX_V2_DTYPE, mode='r',
                    offset=_INDEX_V2_HEADER_SIZE)
    return arr['pid'], arr['offset']


# --- Section parsing and link extraction ---

_RE_WIKILINK = re.compile(r'\[\[([^\[\]|]+)(?:\|[^\[\]]*)?\]\]')
_RE_SECTION_SPLIT = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.MULTILINE)


@dataclass
class LinkWithContext:
    """A link target with surrounding context from the article text."""
    title: str         # normalized link target
    context: str       # ~10-15 words around the link


@dataclass
class ArticleSection:
    """A single section of a parsed article."""
    name: str                    # "Lead", "History", "Geography", etc.
    level: int                   # 0 for lead, 2-6 for headings
    text: str                    # full section text
    links: list[LinkWithContext] # [[link targets]] with context


@dataclass
class ParsedArticle:
    """Article split into infobox + sections with per-section links."""
    infobox: str                    # infobox key-value block (or empty)
    sections: list[ArticleSection]  # lead first, then in document order
    full_text: str                  # original full article text (for search_page)


def parse_article_sections(article_text: str) -> ParsedArticle:
    """Parse cleaned article text into sections with per-section link lists.

    The input is the output of get_article_text(): cleaned wikitext with
    == Section Headers == and [[wikilinks]] preserved.
    """
    if not article_text:
        return ParsedArticle(infobox='', sections=[], full_text='')

    # Separate infobox (appears as [Infobox: ...] block at the top)
    infobox = ''
    body = article_text
    if article_text.startswith('[Infobox:'):
        # Find end of infobox block (double newline after the key-value pairs)
        infobox_end = article_text.find('\n\n')
        if infobox_end != -1:
            infobox = article_text[:infobox_end].strip()
            body = article_text[infobox_end:].strip()
        else:
            infobox = article_text
            body = ''

    # Split body into sections by heading lines
    sections: list[ArticleSection] = []
    # Find all heading positions
    headings = list(_RE_SECTION_SPLIT.finditer(body))

    if not headings:
        # No headings — entire body is the lead
        links = _extract_link_targets(body)
        sections.append(ArticleSection(name='Lead', level=0, text=body, links=links))
    else:
        # Lead = text before first heading
        lead_text = body[:headings[0].start()].strip()
        if lead_text:
            lead_links = _extract_link_targets(lead_text)
            sections.append(ArticleSection(
                name='Lead', level=0, text=lead_text, links=lead_links))

        # Each heading section
        for i, m in enumerate(headings):
            level = len(m.group(1))
            name = m.group(2).strip()
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
            sec_text = body[start:end].strip()
            sec_links = _extract_link_targets(sec_text)
            sections.append(ArticleSection(
                name=name, level=level, text=sec_text, links=sec_links))

    # Also extract links from the infobox
    if infobox:
        infobox_links = _extract_link_targets(infobox)
        if infobox_links:
            sections.insert(0, ArticleSection(
                name='Infobox', level=0, text=infobox, links=infobox_links))

    return ParsedArticle(infobox=infobox, sections=sections, full_text=article_text)


def _extract_link_targets(text: str) -> list[LinkWithContext]:
    """Extract unique [[link targets]] with surrounding context."""
    seen: set[str] = set()
    result: list[LinkWithContext] = []
    for m in _RE_WIKILINK.finditer(text):
        target = m.group(1).strip()
        # Normalize: first char uppercase, spaces to underscores
        target = target.replace(' ', '_')
        if target and target[0].islower():
            target = target[0].upper() + target[1:]
        if not target or target in seen:
            continue
        seen.add(target)
        context = _extract_context(text, m.start(), m.end())
        result.append(LinkWithContext(title=target, context=context))
    return result


_CONTEXT_CHARS = 80  # chars of context on each side of the link


def _extract_context(text: str, start: int, end: int) -> str:
    """Extract a context window around a [[link]] in the raw text.

    Returns ~10-15 words surrounding the link, with the link's display text
    inline (stripped of [[ ]] markup).
    """
    # Grab a window around the match
    ctx_start = max(0, start - _CONTEXT_CHARS)
    ctx_end = min(len(text), end + _CONTEXT_CHARS)

    # Expand to word boundaries
    if ctx_start > 0:
        sp = text.rfind(' ', ctx_start - 20, ctx_start + 10)
        if sp != -1:
            ctx_start = sp + 1

    if ctx_end < len(text):
        sp = text.find(' ', ctx_end - 10, ctx_end + 20)
        if sp != -1:
            ctx_end = sp

    snippet = text[ctx_start:ctx_end]
    # Strip [[ ]] markup from all links in the snippet for readability
    snippet = _RE_WIKILINK.sub(
        lambda m2: m2.group(0).split('|')[-1].rstrip(']]') if '|' in m2.group(0)
        else m2.group(1),
        snippet)
    snippet = snippet.strip()

    # Add ellipsis if truncated
    if ctx_start > 0:
        snippet = '...' + snippet
    if ctx_end < len(text):
        snippet = snippet + '...'
    return snippet
