"""Import Wikipedia SQL dumps into a SQLite database."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import click

from wikipedia_bench.db import (
    create_post_import_indexes,
    get_meta,
    import_pagelinks_fast,
    import_table,
    init_schema,
    open_db,
    set_meta,
)
from wikipedia_bench.redirects import build_canonical_table, build_redirect_map

# Import steps in dependency order.
# Each entry: (step_name, dump_file_glob, table_key_in_TABLE_SPECS)
IMPORT_STEPS = [
    ('page', 'enwiki-latest-page.sql.gz', 'page'),
    ('linktarget', 'enwiki-latest-linktarget.sql.gz', 'linktarget'),
    ('pagelinks', 'enwiki-latest-pagelinks.sql.gz', 'pagelinks'),
    ('redirect', 'enwiki-latest-redirect.sql.gz', 'redirect'),
    ('page_props', 'enwiki-latest-page_props.sql.gz', 'page_props'),
]

DERIVED_STEPS = ['canonical', 'redirect_map']


def _find_file(raw_dir: Path, pattern: str) -> Path | None:
    matches = list(raw_dir.glob(pattern))
    return matches[0] if matches else None


@click.command()
@click.option('--raw-dir', default='data/raw', type=click.Path(exists=True),
              help='Directory containing downloaded dump files')
@click.option('--db-path', default='data/processed/wiki.db', type=click.Path(),
              help='Output SQLite database path')
@click.option('--step', type=click.Choice(
    [s[0] for s in IMPORT_STEPS] + DERIVED_STEPS + ['all'],
), default='all', help='Run a specific step (default: all)')
@click.option('--force/--no-force', default=False,
              help='Re-run steps even if already completed')
def main(raw_dir: str, db_path: str, step: str, force: bool) -> None:
    """Import SQL dumps into SQLite and build derived tables."""
    raw = Path(raw_dir)
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = open_db(db, bulk_mode=True)
    init_schema(conn)

    steps_to_run = []
    if step == 'all':
        steps_to_run = [s[0] for s in IMPORT_STEPS] + ['indexes'] + DERIVED_STEPS
    else:
        steps_to_run = [step]

    for step_name in steps_to_run:
        if step_name == 'indexes':
            t0 = time.time()
            create_post_import_indexes(conn)
            print(f"  Indexes created in {time.time() - t0:.1f}s")
            continue
        meta_key = f'step_{step_name}_done'
        if not force and get_meta(conn, meta_key):
            print(f"Skipping {step_name} (already done). Use --force to re-run.")
            continue

        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Step: {step_name}")
        print(f"{'='*60}")

        if step_name in DERIVED_STEPS:
            _run_derived_step(conn, step_name)
        else:
            spec = next(s for s in IMPORT_STEPS if s[0] == step_name)
            _, filename, table_key = spec
            dump_path = _find_file(raw, filename)
            if dump_path is None:
                print(f"  WARNING: {filename} not found in {raw}, skipping")
                continue
            print(f"  Importing {dump_path.name}...")
            if table_key == 'pagelinks':
                count = import_pagelinks_fast(conn, dump_path)
            else:
                count = import_table(conn, dump_path, table_key)
            print(f"  Imported {count:,} rows")
            set_meta(conn, f'n_{step_name}_rows', str(count))

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        set_meta(conn, meta_key, datetime.now(timezone.utc).isoformat())

    conn.close()
    print(f"\nDatabase: {db}")
    print("Done.")


def _run_derived_step(conn, step_name: str) -> None:
    if step_name == 'canonical':
        count = build_canonical_table(conn)
        print(f"  {count:,} canonical articles")
    elif step_name == 'redirect_map':
        count = build_redirect_map(conn)
        print(f"  {count:,} redirects resolved")


if __name__ == '__main__':
    main()
