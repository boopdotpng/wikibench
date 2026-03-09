"""Build CSR graph from wiki.db."""

from __future__ import annotations

import time
from pathlib import Path

import click

from wikipedia_bench.db import open_db, set_meta
from wikipedia_bench.graph import build_graph


@click.command()
@click.option('--db-path', default='data/processed/wiki.db',
              type=click.Path(exists=True), help='SQLite database path')
@click.option('--out-dir', default='data/processed',
              type=click.Path(), help='Output directory for graph files')
def main(db_path: str, out_dir: str) -> None:
    """Construct CSR directed graph from imported wiki data."""
    conn = open_db(Path(db_path))
    out = Path(out_dir)

    t0 = time.time()
    graph = build_graph(conn, out)
    elapsed = time.time() - t0

    print(f"\nGraph built in {elapsed:.1f}s")
    print(f"  Nodes: {graph.n_nodes:,}")
    print(f"  Forward edges: {graph.n_fwd_edges:,}")
    print(f"  Backward edges: {graph.n_bwd_edges:,}")
    print(f"  Files saved to {out}/")

    conn.close()


if __name__ == '__main__':
    main()
