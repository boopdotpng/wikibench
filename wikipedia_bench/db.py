"""SQLite database schema and bulk import utilities for wiki dump data."""

from __future__ import annotations

import gzip
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Iterator

from .sql_parser import _ProgressWrapper, iter_sql_rows

SCHEMA = """\
-- Raw imports from dump files --

CREATE TABLE IF NOT EXISTS page (
    page_id      INTEGER PRIMARY KEY,
    page_ns      INTEGER NOT NULL,
    page_title   TEXT NOT NULL,
    is_redirect  INTEGER NOT NULL DEFAULT 0,
    page_len     INTEGER
);

CREATE TABLE IF NOT EXISTS redirect_raw (
    rd_from      INTEGER PRIMARY KEY,
    rd_ns        INTEGER NOT NULL,
    rd_title     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS linktarget (
    lt_id        INTEGER PRIMARY KEY,
    lt_ns        INTEGER NOT NULL,
    lt_title     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pagelinks_raw (
    pl_from      INTEGER NOT NULL,
    pl_target_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS page_props (
    pp_page      INTEGER NOT NULL,
    pp_propname  TEXT NOT NULL,
    pp_value     TEXT,
    PRIMARY KEY (pp_page, pp_propname)
);

-- Derived / materialized tables --

CREATE TABLE IF NOT EXISTS canonical (
    page_id      INTEGER PRIMARY KEY,
    page_title   TEXT NOT NULL UNIQUE,
    node_idx     INTEGER NOT NULL UNIQUE,
    out_degree   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS redirect_map (
    rd_from_id   INTEGER PRIMARY KEY,
    canonical_id INTEGER
);

CREATE TABLE IF NOT EXISTS build_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Indexes created AFTER bulk import for speed
POST_IMPORT_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_page_ns_title ON page(page_ns, page_title);
CREATE INDEX IF NOT EXISTS idx_pl_from ON pagelinks_raw(pl_from);
CREATE INDEX IF NOT EXISTS idx_pl_target ON pagelinks_raw(pl_target_id);
CREATE INDEX IF NOT EXISTS idx_canonical_idx ON canonical(node_idx);
"""

# Column indices for each dump table's INSERT VALUES.
# MediaWiki schema as of MW 1.43+ (2026 enwiki/latest).
#
# page: page_id(0), page_namespace(1), page_title(2), page_is_redirect(3),
#        page_is_new(4), page_random(5), page_touched(6), page_links_updated(7),
#        page_latest(8), page_len(9), page_content_model(10), page_lang(11)
#
# redirect: rd_from(0), rd_namespace(1), rd_title(2), rd_interwiki(3), rd_fragment(4)
#
# linktarget: lt_id(0), lt_namespace(1), lt_title(2)
#
# pagelinks: pl_from(0), pl_from_namespace(1), pl_target_id(2)
#
# page_props: pp_page(0), pp_propname(1), pp_value(2), pp_sortkey(3)


def _tx_page(row: list) -> tuple | None:
    """Transform a page dump row into (page_id, page_ns, page_title, is_redirect, page_len)."""
    try:
        return (int(row[0]), int(row[1]), row[2], int(row[3]), int(row[4]) if row[4] is not None else 0)
    except (IndexError, TypeError, ValueError):
        return None


def _tx_redirect(row: list) -> tuple | None:
    """Transform a redirect dump row. Skip interwiki redirects."""
    try:
        interwiki = row[3] if len(row) > 3 else None
        if interwiki is not None and interwiki != '' and interwiki != 'NULL':
            return None
        return (int(row[0]), int(row[1]), row[2])
    except (IndexError, TypeError, ValueError):
        return None


def _tx_linktarget(row: list) -> tuple | None:
    try:
        return (int(row[0]), int(row[1]), row[2])
    except (IndexError, TypeError, ValueError):
        return None


def _tx_pagelinks(row: list) -> tuple | None:
    """Transform pagelinks row. Keep only links from namespace 0."""
    try:
        from_ns = int(row[1])
        if from_ns != 0:
            return None
        return (int(row[0]), int(row[2]))
    except (IndexError, TypeError, ValueError):
        return None


def _tx_page_props(row: list) -> tuple | None:
    """Transform page_props row. Keep only disambiguation markers."""
    try:
        propname = row[1]
        if propname != 'disambiguation':
            return None
        return (int(row[0]), propname, row[2] if len(row) > 2 else '')
    except (IndexError, TypeError, ValueError):
        return None


# Registry:
#   dump_table_name,
#   db_table_name,
#   db_columns,
#   selected_dump_columns,
#   transform_fn,
#   batch_size
TABLE_SPECS = {
    'page': (
        'page',
        'page',
        ['page_id', 'page_ns', 'page_title', 'is_redirect', 'page_len'],
        [0, 1, 2, 3, 9],
        _tx_page,
        100_000,
    ),
    'redirect': (
        'redirect',
        'redirect_raw',
        ['rd_from', 'rd_ns', 'rd_title'],
        [0, 1, 2, 3],
        _tx_redirect,
        100_000,
    ),
    'linktarget': (
        'linktarget',
        'linktarget',
        ['lt_id', 'lt_ns', 'lt_title'],
        [0, 1, 2],
        _tx_linktarget,
        200_000,
    ),
    'pagelinks': (
        'pagelinks',
        'pagelinks_raw',
        ['pl_from', 'pl_target_id'],
        [0, 1, 2],
        _tx_pagelinks,
        500_000,
    ),
    'page_props': (
        'page_props',
        'page_props',
        ['pp_page', 'pp_propname', 'pp_value'],
        [0, 1, 2],
        _tx_page_props,
        100_000,
    ),
}


def open_db(path: Path, *, readonly: bool = False, bulk_mode: bool = False) -> sqlite3.Connection:
    """Open (or create) the wiki.db with tuned pragmas.

    bulk_mode: use journal_mode=OFF and synchronous=OFF for fastest bulk import.
    Not crash-safe, but we can re-run if it fails.
    """
    if readonly:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=2147483648")  # 2 GB mmap
    if bulk_mode:
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA cache_size=-256000")  # 256 MB cache (conservative for 32GB systems)
        conn.execute("PRAGMA cache_spill=OFF")
    elif not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-512000")  # 512 MB cache
    conn.row_factory = sqlite3.Row
    return conn


def create_post_import_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes after bulk import is complete."""
    print("Creating indexes...")
    conn.executescript(POST_IMPORT_INDEXES)
    print("Indexes created.")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def bulk_insert(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: Iterable[tuple],
    batch_size: int = 200_000,
) -> int:
    """Insert rows in batches. Returns total row count."""
    placeholders = ','.join('?' for _ in columns)
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    total = 0
    batch: list[tuple] = []
    cursor = conn.cursor()
    t0 = time.time()
    last_report = t0
    cursor.execute("BEGIN")
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                cursor.executemany(sql, batch)
                total += len(batch)
                batch.clear()
                now = time.time()
                if now - last_report >= 5.0:
                    elapsed = now - t0
                    rate = total / elapsed if elapsed > 0 else 0
                    print(f"    {table}: {total:,} rows ({rate:,.0f} rows/s)")
                    last_report = now
        if batch:
            cursor.executemany(sql, batch)
            total += len(batch)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"    {table}: {total:,} rows total in {elapsed:.1f}s ({rate:,.0f} rows/s)")
    return total


_PAGELINKS_NS0_RE = re.compile(rb'\((\d+),0,(\d+)\)')


def import_pagelinks_fast(
    conn: sqlite3.Connection,
    sql_path: Path,
    *,
    progress: bool = True,
    chunk_size: int = 16 * 1024 * 1024,
    batch_size: int = 500_000,
) -> int:
    """Fast-path import for pagelinks dumps with integer-only rows.

    Extracts only `(pl_from, 0, pl_target_id)` tuples using a compiled bytes
    regex over large decompressed chunks, then bulk-inserts `(pl_from,
    pl_target_id)` into `pagelinks_raw`.
    """
    from tqdm import tqdm

    sql_path = Path(sql_path)

    sql = "INSERT OR IGNORE INTO pagelinks_raw (pl_from, pl_target_id) VALUES (?, ?)"
    total = 0
    batch: list[tuple[int, int]] = []
    carry = b''
    cursor = conn.cursor()
    started = time.time()
    last_update = started

    pbar = tqdm(
        total=sql_path.stat().st_size if progress else None,
        unit='B' if progress else 'row',
        unit_scale=progress,
        desc='pagelinks',
        disable=not progress,
    )

    def _flush_batch() -> None:
        nonlocal total
        if not batch:
            return
        cursor.executemany(sql, batch)
        total += len(batch)
        batch.clear()

    def _report_progress(force: bool = False) -> None:
        nonlocal last_update
        now = time.time()
        if not force and now - last_update < 1.0:
            return
        elapsed = now - started
        rows_per_sec = total / elapsed if elapsed > 0 else 0.0
        if progress:
            pbar.set_postfix(rows=total, rows_per_s=f"{rows_per_sec:,.0f}")
        else:
            print(f"    pagelinks_raw: {total:,} rows ({rows_per_sec:,.0f} rows/s)")
        last_update = now

    if sql_path.suffix == '.gz':
        raw_f = open(sql_path, 'rb')
        wrapped = _ProgressWrapper(raw_f, pbar) if progress else raw_f
        f = gzip.open(wrapped, 'rb')  # type: ignore[arg-type]
    else:
        raw_f = None
        f = open(sql_path, 'rb')

    conn.execute("PRAGMA synchronous=OFF")
    cursor.execute("BEGIN")
    try:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            data = carry + chunk
            last_close = data.rfind(b')')
            if last_close < 0:
                carry = data
                continue

            scan_upto = last_close + 1
            carry = data[scan_upto:]
            for match in _PAGELINKS_NS0_RE.finditer(data, 0, scan_upto):
                batch.append((int(match.group(1)), int(match.group(2))))
                if len(batch) >= batch_size:
                    _flush_batch()
                    _report_progress()

        if carry:
            for match in _PAGELINKS_NS0_RE.finditer(carry):
                batch.append((int(match.group(1)), int(match.group(2))))
                if len(batch) >= batch_size:
                    _flush_batch()
                    _report_progress()

        _flush_batch()
        _report_progress(force=True)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA synchronous=NORMAL")
        f.close()
        if raw_f is not None:
            raw_f.close()
        pbar.close()

    elapsed = time.time() - started
    rate = total / elapsed if elapsed > 0 else 0
    print(f"    pagelinks_raw: {total:,} rows total in {elapsed:.1f}s ({rate:,.0f} rows/s)")
    return total


def import_table(
    conn: sqlite3.Connection,
    sql_path: Path,
    table_key: str,
    *,
    progress: bool = True,
) -> int:
    """Import a dump file into the database using the registered spec."""
    dump_table, db_table, columns, selected_columns, transform, batch_size = TABLE_SPECS[table_key]

    def _transformed_rows() -> Iterator[tuple]:
        for raw_row in iter_sql_rows(
            sql_path,
            dump_table,
            progress=progress,
            keep_columns=selected_columns,
        ):
            result = transform(raw_row)
            if result is not None:
                yield result

    # Disable sync for bulk import speed
    conn.execute("PRAGMA synchronous=OFF")
    try:
        count = bulk_insert(conn, db_table, columns, _transformed_rows(), batch_size=batch_size)
    finally:
        conn.execute("PRAGMA synchronous=NORMAL")
    return count


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM build_meta WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None
