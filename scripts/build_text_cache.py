"""Build cached lead text and link lists from the article dump."""

from __future__ import annotations

import time
from pathlib import Path

import click

from wikipedia_bench.db import open_db, set_meta
from wikipedia_bench.text_extract import build_text_cache


@click.command()
@click.option('--db-path', default='data/processed/wiki.db',
              type=click.Path(exists=True), help='SQLite database path')
@click.option('--dump-path', default='data/raw/enwiki-latest-pages-articles-multistream.xml.bz2',
              type=click.Path(exists=True), help='Multistream article dump')
@click.option('--index-path', default='data/raw/enwiki-latest-pages-articles-multistream-index.txt.bz2',
              type=click.Path(exists=True), help='Multistream index file')
@click.option('--workers', default=0, type=int,
              help='Worker processes for block extraction (default: cpu_count - 1)')
@click.option('--batch-size', default=10_000, type=int,
              help='SQLite flush batch size for extracted pages')
def main(db_path: str, dump_path: str, index_path: str, workers: int, batch_size: int) -> None:
    """Extract lead text and visible links for all canonical articles."""
    conn = open_db(Path(db_path))

    t0 = time.time()
    count = build_text_cache(
        Path(dump_path),
        Path(index_path),
        conn,
        workers=workers,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0

    print(f"\nProcessed {count:,} pages in {elapsed:.1f}s")
    set_meta(conn, 'text_cache_built', str(count))
    conn.close()


if __name__ == '__main__':
    main()
