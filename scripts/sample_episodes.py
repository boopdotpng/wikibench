"""Sample benchmark episodes from the frozen graph."""

from __future__ import annotations

import time
from pathlib import Path

import click

from wikipedia_bench.db import open_db
from wikipedia_bench.graph import CSRGraph
from wikipedia_bench.sampler import sample_episodes


@click.command()
@click.option('--db-path', default='data/processed/wiki.db',
              type=click.Path(exists=True), help='SQLite database path')
@click.option('--graph-dir', default='data/processed',
              type=click.Path(exists=True), help='Graph files directory')
@click.option('--episodes-dir', default='episodes',
              type=click.Path(), help='Output directory for episode JSONL files')
@click.option('--n-dev', default=500, help='Number of dev episodes')
@click.option('--n-test', default=2000, help='Number of test episodes')
@click.option('--seed', default=42, help='Base RNG seed')
@click.option('--hub-threshold', default=50_000,
              help='Out-degree threshold for hub classification')
def main(
    db_path: str,
    graph_dir: str,
    episodes_dir: str,
    n_dev: int,
    n_test: int,
    seed: int,
    hub_threshold: int,
) -> None:
    """Sample episodes with difficulty bucketing and write JSONL splits."""
    conn = open_db(Path(db_path), readonly=True)

    print("Loading graph...")
    t0 = time.time()
    graph = CSRGraph.load(Path(graph_dir))
    print(f"  Loaded in {time.time() - t0:.1f}s "
          f"({graph.n_nodes:,} nodes, {graph.n_fwd_edges:,} edges)")

    t0 = time.time()
    results = sample_episodes(
        graph,
        conn,
        n_dev=n_dev,
        n_test=n_test,
        seed=seed,
        hub_threshold=hub_threshold,
        episodes_dir=Path(episodes_dir),
    )
    elapsed = time.time() - t0

    total = sum(len(v) for v in results.values())
    print(f"\nSampled {total} total episodes in {elapsed:.1f}s")

    conn.close()


if __name__ == '__main__':
    main()
