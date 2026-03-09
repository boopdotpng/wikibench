"""Redirect chain resolution and canonical article table construction."""

from __future__ import annotations

import sqlite3
from typing import TextIO

from .db import set_meta


def build_canonical_table(conn: sqlite3.Connection) -> int:
    """Populate the canonical table with non-redirect, non-disambiguation ns0 pages.

    Assigns deterministic sequential node_idx values ordered by page_id.
    Returns the number of canonical articles.
    """
    conn.execute("DELETE FROM canonical")

    conn.execute("""\
        INSERT INTO canonical (page_id, page_title, node_idx, out_degree)
        SELECT
            p.page_id,
            p.page_title,
            (ROW_NUMBER() OVER (ORDER BY p.page_id)) - 1,
            0
        FROM page p
        WHERE p.page_ns = 0
          AND p.is_redirect = 0
          AND p.page_id NOT IN (
              SELECT pp_page FROM page_props WHERE pp_propname = 'disambiguation'
          )
        ORDER BY p.page_id
    """)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM canonical").fetchone()[0]
    set_meta(conn, 'n_canonical', str(count))
    return count


def build_redirect_map(conn: sqlite3.Connection) -> int:
    """Resolve redirect chains and populate redirect_map.

    For each redirect page, follows the chain until reaching a canonical
    (non-redirect) page or detecting a cycle/broken link.

    Returns the number of redirect entries resolved.
    """
    conn.execute("DELETE FROM redirect_map")

    # Load the raw redirect targets: rd_from -> (rd_ns, rd_title)
    raw_redirects: dict[int, tuple[int, str]] = {}
    for row in conn.execute("SELECT rd_from, rd_ns, rd_title FROM redirect_raw"):
        raw_redirects[row[0]] = (row[1], row[2])

    # Build title -> page_id lookup for ns0 pages
    title_to_id: dict[str, int] = {}
    for row in conn.execute("SELECT page_id, page_title FROM page WHERE page_ns = 0"):
        title_to_id[row[1]] = row[0]

    # Set of page_ids that are redirects
    redirect_ids: set[int] = set(raw_redirects.keys())

    # Resolve chains iteratively with path compression
    resolved: dict[int, int | None] = {}

    for start_id in raw_redirects:
        if start_id in resolved:
            continue

        # Walk the chain
        chain: list[int] = []
        visited: set[int] = set()
        cur = start_id

        while True:
            if cur in resolved:
                # Already resolved this node
                target = resolved[cur]
                break

            if cur in visited:
                # Cycle detected
                target = None
                break

            visited.add(cur)
            chain.append(cur)

            if cur not in redirect_ids:
                # cur is not a redirect -> it's the canonical target
                target = cur
                break

            # Follow the redirect
            rd_ns, rd_title = raw_redirects[cur]
            if rd_ns != 0:
                # Redirect points to non-article namespace
                target = None
                break

            next_id = title_to_id.get(rd_title)
            if next_id is None:
                # Broken redirect (target page doesn't exist)
                target = None
                break

            cur = next_id

        # Path compression: all nodes in the chain resolve to the same target
        for node in chain:
            resolved[node] = target

    # Bulk insert into redirect_map
    conn.commit()  # close any open transaction before pragma change
    conn.execute("PRAGMA synchronous=OFF")
    batch: list[tuple[int, int | None]] = []
    for from_id, canon_id in resolved.items():
        batch.append((from_id, canon_id))
        if len(batch) >= 50_000:
            conn.executemany(
                "INSERT OR IGNORE INTO redirect_map (rd_from_id, canonical_id) VALUES (?, ?)",
                batch,
            )
            batch.clear()
    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO redirect_map (rd_from_id, canonical_id) VALUES (?, ?)",
            batch,
        )
    conn.commit()
    conn.execute("PRAGMA synchronous=NORMAL")

    count = len(resolved)
    set_meta(conn, 'n_redirects_resolved', str(count))
    return count
