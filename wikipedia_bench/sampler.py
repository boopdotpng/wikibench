"""Episode sampler for the Wikipedia racing benchmark."""

from __future__ import annotations

import random
import sqlite3
import uuid
from pathlib import Path

from .graph import CSRGraph, bidirectional_bfs
from .schemas import Episode, write_episodes_jsonl


def difficulty_bucket(shortest_path: int) -> str:
    if shortest_path <= 3:
        return 'easy'
    if shortest_path <= 5:
        return 'medium'
    return 'hard'


def step_limit(shortest_path: int) -> int:
    return max(2 * shortest_path + 4, 12)


def sample_episodes(
    graph: CSRGraph,
    conn: sqlite3.Connection,
    *,
    n_dev: int = 500,
    n_test: int = 2000,
    seed: int = 42,
    max_sample_depth: int = 6,
    hub_threshold: int = 50_000,
    episodes_dir: Path = Path('episodes'),
    progress: bool = True,
) -> dict[str, list[Episode]]:
    """Sample benchmark episodes — take any reachable pair, no bucket quotas.

    Fast: most random pairs on Wikipedia are 2-4 hops apart, so nearly every
    attempt succeeds.  Finishes in seconds for typical counts.
    """
    if progress:
        from tqdm import tqdm

    # Load node_idx -> (page_id, page_title) mapping
    idx_to_info: dict[int, tuple[int, str]] = {}
    for row in conn.execute("SELECT page_id, page_title, node_idx FROM canonical"):
        idx_to_info[row[2]] = (row[0], row[1])

    all_indices = list(idx_to_info.keys())

    # Identify hubs by out-degree
    hub_indices: set[int] = set()
    for idx in all_indices:
        if graph.out_degree(idx) > hub_threshold:
            hub_indices.add(idx)

    non_hub_indices = [i for i in all_indices if i not in hub_indices]
    print(f"Candidate pool: {len(all_indices):,} total, {len(hub_indices):,} hubs, "
          f"{len(non_hub_indices):,} non-hubs")

    splits_config = [
        ('dev', n_dev, seed, all_indices),
        ('test', n_test, seed + 1, all_indices),
        ('dev_hub_filtered', n_dev, seed + 2, non_hub_indices),
        ('test_hub_filtered', n_test, seed + 3, non_hub_indices),
    ]

    results: dict[str, list[Episode]] = {}

    for split_name, n_total, split_seed, candidate_pool in splits_config:
        print(f"\nSampling split: {split_name} ({n_total} episodes)")
        rng = random.Random(split_seed)

        episodes: list[Episode] = []
        seen_pairs: set[tuple[int, int]] = set()
        bucket_counts: dict[str, int] = {'easy': 0, 'medium': 0, 'hard': 0}
        attempts = 0
        max_attempts = n_total * 10

        pbar = tqdm(total=n_total, desc=split_name) if progress else None

        while len(episodes) < n_total and attempts < max_attempts:
            attempts += 1

            src_idx = rng.choice(candidate_pool)
            dst_idx = rng.choice(candidate_pool)

            if src_idx == dst_idx:
                continue
            if (src_idx, dst_idx) in seen_pairs:
                continue
            seen_pairs.add((src_idx, dst_idx))

            # Cap BFS at depth 3 — keeps each query very fast
            dist = bidirectional_bfs(graph, src_idx, dst_idx, max_depth=3)
            if dist is None:
                continue

            bucket = difficulty_bucket(dist)
            src_pid, src_title = idx_to_info[src_idx]
            dst_pid, dst_title = idx_to_info[dst_idx]

            ep = Episode(
                episode_id=uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"wikibench.{split_name}.{src_pid}.{dst_pid}.{split_seed}",
                ).hex,
                split=split_name,
                seed=len(episodes),
                start_page_id=src_pid,
                target_page_id=dst_pid,
                start_title=src_title,
                target_title=dst_title,
                shortest_path_len=dist,
                difficulty=bucket,
                step_limit=step_limit(dist),
            )
            episodes.append(ep)
            bucket_counts[bucket] += 1

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        episodes.sort(key=lambda e: e.seed)

        print(f"  Sampled {len(episodes)} episodes in {attempts} attempts")
        for b in ['easy', 'medium', 'hard']:
            print(f"    {b}: {bucket_counts[b]}")

        results[split_name] = episodes

        out_path = episodes_dir / f"{split_name}.jsonl"
        write_episodes_jsonl(episodes, out_path)
        print(f"  Written to {out_path}")

    return results
