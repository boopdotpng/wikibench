"""Sample a single benchmark episode set from the frozen graph."""

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
@click.option('--episodes-path', default='episodes/benchmark.jsonl',
              type=click.Path(), help='Output JSONL path for the benchmark episodes (writes a sibling CSV too)')
@click.option('--hard-bonus-path', default='episodes/hard_bonus.jsonl',
              type=click.Path(), help='Output JSONL path for the hard bonus episodes (writes a sibling CSV too)')
@click.option('--n-total', default=100, help='Total number of benchmark episodes')
@click.option('--n-hard-bonus', default=3, help='Number of hard bonus episodes at depth 6-7')
@click.option('--seed', default=42, help='Base RNG seed')
@click.option('--max-sample-depth', default=7, help='Maximum shortest-path depth to verify')
@click.option('--hard-source-max-out-degree', default=20,
              help='Prefer low out-degree pages as hard-example sources')
def main(
    db_path: str,
    graph_dir: str,
    episodes_path: str,
    hard_bonus_path: str,
    n_total: int,
    n_hard_bonus: int,
    seed: int,
    max_sample_depth: int,
    hard_source_max_out_degree: int,
) -> None:
    """Sample a single benchmark set and write it to JSONL."""
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
        n_total=n_total,
        n_hard_bonus=n_hard_bonus,
        seed=seed,
        max_sample_depth=max_sample_depth,
        hard_source_max_out_degree=hard_source_max_out_degree,
        episodes_path=Path(episodes_path),
        hard_bonus_path=Path(hard_bonus_path),
    )
    elapsed = time.time() - t0

    print(
        f"\nSampled {len(results['benchmark'])} benchmark episodes and "
        f"{len(results['hard_bonus'])} hard bonus episodes in {elapsed:.1f}s"
    )
    conn.close()


if __name__ == '__main__':
    main()
