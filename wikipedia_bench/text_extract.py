"""Lead text extraction from Wikipedia multistream dump.

Uses the multistream index for efficient random access, and mwparserfromhell
for wikitext-to-plaintext conversion.
"""

from __future__ import annotations

import bz2
import io
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

# Lead text limits
MAX_LEAD_CHARS = 4000  # roughly 800-1500 tokens
MAX_LEAD_PARAGRAPHS = 3

# Patterns to strip from wikitext before parsing
_RE_REF = re.compile(r'<ref[^>]*/>', re.DOTALL)
_RE_REF_BLOCK = re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL)
_RE_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)
_RE_GALLERY = re.compile(r'<gallery[^>]*>.*?</gallery>', re.DOTALL)
_RE_MATH = re.compile(r'<math[^>]*>.*?</math>', re.DOTALL)
_RE_NOWIKI = re.compile(r'<nowiki[^>]*>.*?</nowiki>', re.DOTALL)

# Post-parse cleanup
_RE_MULTI_NEWLINE = re.compile(r'\n{3,}')
_RE_MULTI_SPACE = re.compile(r'  +')
_RE_PAREN_EMPTY = re.compile(r'\(\s*\)')
_RE_BRACKET_EMPTY = re.compile(r'\[\s*\]')


def parse_multistream_index(index_path: Path) -> dict[str, tuple[int, int]]:
    """Parse the multistream index file.

    Returns {page_title: (byte_offset, page_id)} for all indexed pages.
    The byte_offset points to the start of the bz2 block in the multistream
    dump that contains this page.
    """
    title_to_loc: dict[str, tuple[int, int]] = {}

    opener = bz2.open if index_path.suffix == '.bz2' else open
    with opener(index_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: byte_offset:page_id:page_title
            parts = line.split(':', 2)
            if len(parts) < 3:
                continue
            try:
                offset = int(parts[0])
                page_id = int(parts[1])
                title = parts[2]
                title_to_loc[title] = (offset, page_id)
            except ValueError:
                continue

    return title_to_loc


def build_offset_index(
    index_path: Path,
    *,
    only_page_ids: set[int] | None = None,
) -> dict[int, int]:
    """Build page_id -> byte_offset mapping from the multistream index.

    Returns {page_id: byte_offset} where byte_offset is the start of the
    bz2 block containing this page in the multistream dump.
    """
    pid_to_offset: dict[int, int] = {}
    remaining = set(only_page_ids) if only_page_ids is not None else None

    opener = bz2.open if index_path.suffix == '.bz2' else open
    with opener(index_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':', 2)
            if len(parts) < 3:
                continue
            try:
                offset = int(parts[0])
                page_id = int(parts[1])
                if remaining is not None and page_id not in remaining:
                    continue
                pid_to_offset[page_id] = offset
                if remaining is not None:
                    remaining.discard(page_id)
                    if not remaining:
                        break
            except ValueError:
                continue

    return pid_to_offset


def read_bz2_block(dump_path: Path, offset: int) -> bytes:
    """Read and decompress a single bz2 block from the multistream dump.

    Each block in the multistream dump is an independent bz2 stream
    containing one or more <page> elements within a wrapper.
    """
    dec = bz2.BZ2Decompressor()
    result = bytearray()

    with open(dump_path, 'rb') as f:
        f.seek(offset)
        while not dec.eof:
            chunk = f.read(1024 * 1024)  # 1MB at a time
            if not chunk:
                break
            result.extend(dec.decompress(chunk))

    return bytes(result)


def _local_tag_name(tag: str) -> str:
    """Strip XML namespace from a tag."""
    return tag.rsplit('}', 1)[-1]


def extract_pages_from_block(block_data: bytes, target_page_ids: set[int]) -> dict[int, str]:
    """Extract wikitext for several page_ids from a decompressed block."""
    if not target_page_ids:
        return {}

    xml_bytes = b'<mediawiki>' + block_data + b'</mediawiki>'
    found: dict[int, str] = {}

    try:
        parser = ET.iterparse(io.BytesIO(xml_bytes), events=('end',))

        for _, elem in parser:
            if _local_tag_name(elem.tag) != 'page':
                continue

            page_id: int | None = None
            page_text: str | None = None

            for child in elem:
                child_name = _local_tag_name(child.tag)
                if child_name == 'id' and page_id is None:
                    try:
                        if child.text is not None:
                            page_id = int(child.text)
                    except ValueError:
                        page_id = None
                elif child_name == 'revision':
                    for rev_child in child:
                        if _local_tag_name(rev_child.tag) == 'text':
                            page_text = rev_child.text if rev_child.text else None
                            break

            if page_id in target_page_ids and page_text is not None:
                found[page_id] = page_text
                if len(found) == len(target_page_ids):
                    elem.clear()
                    break

            elem.clear()
    except ET.ParseError:
        return found

    return found


def extract_page_text_from_block(block_data: bytes, target_page_id: int) -> str | None:
    """Extract the wikitext for a specific page_id from a decompressed block.

    The block contains XML fragments with <page> elements.
    """
    return extract_pages_from_block(block_data, {target_page_id}).get(target_page_id)


def extract_lead_text(wikitext: str) -> str:
    """Extract clean lead text from raw wikitext.

    Returns the first few paragraphs before the first section heading,
    cleaned of templates, references, and markup.
    """
    # Find the lead section (before first ==heading==)
    heading_match = re.search(r'\n==[^=]', wikitext)
    if heading_match:
        lead_wikitext = wikitext[:heading_match.start()]
    else:
        lead_wikitext = wikitext

    # Strip HTML-like tags we don't want
    lead_wikitext = _RE_REF.sub('', lead_wikitext)
    lead_wikitext = _RE_REF_BLOCK.sub('', lead_wikitext)
    lead_wikitext = _RE_COMMENT.sub('', lead_wikitext)
    lead_wikitext = _RE_GALLERY.sub('', lead_wikitext)
    lead_wikitext = _RE_MATH.sub('', lead_wikitext)
    lead_wikitext = _RE_NOWIKI.sub('', lead_wikitext)

    # Use fast regex-based stripping (mwparserfromhell is too slow for bulk)
    text = _regex_strip_wikitext(lead_wikitext)

    # Post-cleanup
    text = _RE_PAREN_EMPTY.sub('', text)
    text = _RE_BRACKET_EMPTY.sub('', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    text = _RE_MULTI_NEWLINE.sub('\n\n', text)
    text = text.strip()

    # Truncate to lead limits
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    paragraphs = paragraphs[:MAX_LEAD_PARAGRAPHS]
    text = '\n\n'.join(paragraphs)

    if len(text) > MAX_LEAD_CHARS:
        text = text[:MAX_LEAD_CHARS].rsplit(' ', 1)[0] + '...'

    return text


_RE_TEMPLATE = re.compile(r'\{\{[^{}]*\}\}')


def _regex_strip_wikitext(text: str) -> str:
    """Fast wikitext stripper using regex."""
    # Strip templates {{...}} iteratively (handles nesting via repeated passes)
    # Most templates are removed in 2-3 passes; cap at 10 to avoid pathological cases
    for _ in range(10):
        new = _RE_TEMPLATE.sub('', text)
        if new == text:
            break
        text = new

    # Convert wikilinks [[target|display]] -> display, [[target]] -> target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)

    # Strip bold/italic markup
    text = text.replace("'''", '').replace("''", '')

    # Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    return text


def extract_links_from_wikitext(wikitext: str) -> list[str]:
    """Extract visible article links from wikitext.

    Implements the human-adjacent visibility filter from spec section 5.3:
    - Include: prose/body links, hatnotes, infobox links, See also
    - Exclude: navboxes, footer nav templates, citations, non-article namespaces

    Returns deduplicated list of canonical article titles (underscored).
    """
    # Find the article body (everything up to navbox/footer templates)
    # Navboxes are typically at the very end, inside {{Navbox or similar
    body = _strip_navbox_region(wikitext)

    # Use fast regex-based extraction (mwparserfromhell is too slow for bulk)
    return _regex_extract_links(body)


def _strip_navbox_region(wikitext: str) -> str:
    """Remove navbox/footer template regions from the end of wikitext.

    Navboxes are typically {{Navbox, {{Authority control}}, {{Taxonbar}},
    {{Portal bar}}, etc. at the very end of the article.
    """
    # Common navbox template names (case-insensitive starts)
    navbox_prefixes = (
        'navbox', 'authority control', 'taxonbar', 'portal bar',
        'portal', 'commons category', 'wikiquote', 'wikisource',
        'wiktionary', 'reflist', 'notelist', 'refbegin', 'refend',
        'cite', 'citation', 'sfn', 'efn', 'cnote',
    )

    lines = wikitext.split('\n')
    cutoff = len(lines)

    # Scan from the end, looking for where navbox templates start
    i = len(lines) - 1
    while i >= 0:
        stripped = lines[i].strip().lower()
        if stripped.startswith('{{'):
            template_name = stripped[2:].split('|')[0].split('}')[0].strip()
            if any(template_name.startswith(p) for p in navbox_prefixes):
                cutoff = i
                i -= 1
                continue
        if stripped == '' or stripped.startswith('[[category:'):
            cutoff = i
            i -= 1
            continue
        break

    return '\n'.join(lines[:cutoff])


def _regex_extract_links(text: str) -> list[str]:
    """Fallback link extraction using regex."""
    seen: set[str] = set()
    links: list[str] = []

    for match in re.finditer(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', text):
        title = match.group(1).strip()
        if ':' in title:
            prefix = title.split(':', 1)[0]
            if prefix in ('File', 'Image', 'Category', 'Template', 'Talk',
                          'Portal', 'Wikipedia', 'Help', 'Draft', 'Module'):
                continue
        title = title.split('#', 1)[0].strip()
        if not title:
            continue
        title = title.replace(' ', '_')
        title = title[0].upper() + title[1:] if title else title
        if title not in seen:
            seen.add(title)
            links.append(title)

    return links


# --- Bulk text cache building ---

LEAD_TEXT_SCHEMA = """\
CREATE TABLE IF NOT EXISTS lead_text (
    page_id   INTEGER PRIMARY KEY,
    lead_text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS page_links (
    page_id   INTEGER NOT NULL,
    link_title TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_page_links_pid ON page_links(page_id);
"""


def _process_block(args: tuple) -> list[tuple[int, str, list[str]]]:
    """Process a single bz2 block in a worker process.

    Returns list of (page_id, lead_text, [link_titles]) for each page found.
    """
    dump_path_str, offset, pids = args
    dump_path = Path(dump_path_str)
    results: list[tuple[int, str, list[str]]] = []

    try:
        block_data = read_bz2_block(dump_path, offset)
    except Exception:
        return results

    block_pages = extract_pages_from_block(block_data, set(pids))
    for pid in pids:
        wikitext = block_pages.get(pid)
        if wikitext is None:
            continue

        lead = extract_lead_text(wikitext)
        page_links = extract_links_from_wikitext(wikitext)
        results.append((pid, lead, page_links))

    return results


def build_text_cache(
    dump_path: Path,
    index_path: Path,
    conn: sqlite3.Connection,
    *,
    progress: bool = True,
    batch_size: int = 10_000,
    workers: int = 0,
) -> int:
    """Build cached lead text and link lists for all canonical articles.

    Reads the multistream dump using the index for block-level access.
    Extracts lead text and visible links for each canonical page.
    Stores results in the lead_text and page_links tables.

    Uses multiprocessing for parallel block processing.

    Returns number of pages processed.
    """
    import multiprocessing as mp
    if progress:
        from tqdm import tqdm

    if workers <= 0:
        workers = max(1, os.cpu_count() - 1)

    conn.executescript(LEAD_TEXT_SCHEMA)

    # Get all canonical page_ids
    canonical_pids: set[int] = set()
    for row in conn.execute("SELECT page_id FROM canonical"):
        canonical_pids.add(row[0])

    # Already cached
    cached: set[int] = set()
    for row in conn.execute("SELECT page_id FROM lead_text"):
        cached.add(row[0])

    to_process = canonical_pids - cached
    if not to_process:
        print("All pages already cached.")
        return 0

    print(f"Building text cache for {len(to_process):,} pages "
          f"({len(cached):,} already cached)")

    # Build page_id -> offset index
    print("Loading multistream index...")
    t_index = time.time()
    pid_to_offset = build_offset_index(index_path, only_page_ids=to_process)
    print(f"  Loaded offsets for {len(pid_to_offset):,} pages in {time.time() - t_index:.1f}s")

    # Group pages by block offset for efficient reading
    offset_to_pids: dict[int, list[int]] = {}
    missing = 0
    for pid in to_process:
        offset = pid_to_offset.get(pid)
        if offset is None:
            missing += 1
            continue
        offset_to_pids.setdefault(offset, []).append(pid)

    if missing:
        print(f"  {missing:,} pages not found in index")

    offsets = sorted(offset_to_pids.keys())
    n_blocks = len(offsets)
    avg_pages_per_block = (len(to_process) - missing) / n_blocks if n_blocks else 0.0
    print(f"  {n_blocks:,} bz2 blocks to read, {workers} workers, {avg_pages_per_block:.1f} pages/block")

    # Build work items
    work_items = [(str(dump_path), offset, offset_to_pids[offset]) for offset in offsets]

    processed = 0
    text_batch: list[tuple[int, str]] = []
    links_batch: list[tuple[int, str]] = []
    chunksize = max(1, min(32, n_blocks // max(1, workers * 4)))
    t0 = time.time()

    conn.execute("PRAGMA synchronous=OFF")

    # Use multiprocessing pool
    try:
        with mp.Pool(workers, maxtasksperchild=512) as pool:
            iterator = pool.imap_unordered(_process_block, work_items, chunksize=chunksize)
            pbar = tqdm(total=len(to_process), desc="pages", unit="page") if progress else None

            for block_results in iterator:
                for pid, lead, page_links in block_results:
                    text_batch.append((pid, lead))
                    for link_title in page_links:
                        links_batch.append((pid, link_title))
                    processed += 1

                if pbar is not None and block_results:
                    pbar.update(len(block_results))
                    elapsed = max(time.time() - t0, 1e-9)
                    pbar.set_postfix_str(f"{processed / elapsed:,.0f} pages/s")

                if len(text_batch) >= batch_size:
                    _flush_batches(conn, text_batch, links_batch)
                    text_batch.clear()
                    links_batch.clear()

        if text_batch:
            _flush_batches(conn, text_batch, links_batch)
    finally:
        if progress and 'pbar' in locals() and pbar is not None:
            pbar.close()
        conn.execute("PRAGMA synchronous=NORMAL")

    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0.0
    print(f"  Processed {processed:,} pages in {elapsed:.1f}s ({rate:,.0f} pages/s)")
    return processed


def _flush_batches(
    conn: sqlite3.Connection,
    text_batch: list[tuple[int, str]],
    links_batch: list[tuple[int, str]],
) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO lead_text (page_id, lead_text) VALUES (?, ?)",
        text_batch,
    )
    conn.executemany(
        "INSERT INTO page_links (page_id, link_title) VALUES (?, ?)",
        links_batch,
    )
    conn.commit()
