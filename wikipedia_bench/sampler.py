"""Episode sampler for the Wikipedia racing benchmark."""

from __future__ import annotations

import json
import random
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .graph import CSRGraph, bidirectional_bfs
from .schemas import Episode, write_episodes_csv, write_episodes_jsonl


def difficulty_bucket(shortest_path: int) -> str:
    if shortest_path <= 4:
        return 'easy'
    if shortest_path == 5:
        return 'medium'
    return 'hard'


def step_limit(shortest_path: int) -> int:
    return max(2 * shortest_path + 4, 12)


def bonus_step_limit(shortest_path: int) -> int:
    return max(3 * shortest_path + 4, 20)


def _default_bucket_targets(n_total: int) -> dict[str, int]:
    easy = round(n_total * 0.4)
    medium = n_total - easy
    return {'easy': easy, 'medium': medium}


def _write_dataset_metadata(
    path: Path,
    episodes: list[Episode],
    *,
    seed: int,
    n_total: int,
    n_hard_bonus: int,
    max_sample_depth: int,
    hard_source_max_out_degree: int,
    hard_walk_max_out_degree: int,
) -> None:
    counts: dict[str, int] = {}
    for episode in episodes:
        counts[episode.difficulty] = counts.get(episode.difficulty, 0) + 1

    metadata = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'seed': seed,
        'n_total': n_total,
        'n_hard_bonus': n_hard_bonus,
        'max_sample_depth': max_sample_depth,
        'hard_source_max_out_degree': hard_source_max_out_degree,
        'hard_walk_max_out_degree': hard_walk_max_out_degree,
        'episode_count': len(episodes),
        'difficulty_counts': counts,
        'data_file': path.name,
    }
    meta_path = path.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(metadata, indent=2))


def _pick_unvisited_neighbor(
    graph: CSRGraph,
    rng: random.Random,
    node_idx: int,
    visited: set[int],
    *,
    max_out_degree: int | None = None,
) -> int | None:
    neighbors = graph.neighbors(node_idx)
    if len(neighbors) == 0:
        return None

    # Fast path: try a few random picks before falling back to a scan.
    for _ in range(min(8, len(neighbors))):
        candidate = int(neighbors[rng.randrange(len(neighbors))])
        if candidate in visited:
            continue
        if max_out_degree is not None and graph.out_degree(candidate) > max_out_degree:
            continue
        return candidate

    for neighbor in neighbors:
        candidate = int(neighbor)
        if candidate in visited:
            continue
        if max_out_degree is not None and graph.out_degree(candidate) > max_out_degree:
            continue
        return candidate

    return None


def _random_walk_target(
    graph: CSRGraph,
    rng: random.Random,
    src_idx: int,
    steps: int,
    *,
    max_neighbor_out_degree: int | None = None,
) -> int | None:
    current = src_idx
    visited = {src_idx}

    for _ in range(steps):
        current = _pick_unvisited_neighbor(
            graph,
            rng,
            current,
            visited,
            max_out_degree=max_neighbor_out_degree,
        )
        if current is None:
            return None
        visited.add(current)

    return current


def _sample_bucket(
    graph: CSRGraph,
    rng: random.Random,
    idx_to_info: dict[int, tuple[int, str]],
    source_pool: list[int],
    *,
    split_name: str,
    target_count: int,
    walk_lengths: tuple[int, ...],
    accepted_distances: set[int],
    verify_depth: int,
    seen_pairs: set[tuple[int, int]],
    start_seed: int,
    max_neighbor_out_degree: int | None = None,
    proposal_mode: str = 'walk',
    progress_cb=None,
) -> tuple[list[Episode], int, int]:
    episodes: list[Episode] = []
    attempts = 0
    max_attempts = max(target_count * 200, 1_000)

    while len(episodes) < target_count and attempts < max_attempts:
        attempts += 1

        src_idx = rng.choice(source_pool)
        if proposal_mode == 'random_pair':
            dst_idx = rng.choice(source_pool)
        else:
            walk_length = rng.choice(walk_lengths)
            dst_idx = _random_walk_target(
                graph,
                rng,
                src_idx,
                walk_length,
                max_neighbor_out_degree=max_neighbor_out_degree,
            )
        if dst_idx is None or dst_idx == src_idx or (src_idx, dst_idx) in seen_pairs:
            continue

        dist = bidirectional_bfs(graph, src_idx, dst_idx, max_depth=verify_depth)
        if dist not in accepted_distances:
            continue

        seen_pairs.add((src_idx, dst_idx))
        src_pid, src_title = idx_to_info[src_idx]
        dst_pid, dst_title = idx_to_info[dst_idx]
        episodes.append(Episode(
            episode_id=uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"wikibench.{split_name}.{src_pid}.{dst_pid}.{start_seed + len(episodes)}",
            ).hex,
            split=split_name,
            seed=start_seed + len(episodes),
            start_page_id=src_pid,
            target_page_id=dst_pid,
            start_title=src_title,
            target_title=dst_title,
            shortest_path_len=dist,
            difficulty=difficulty_bucket(dist),
            step_limit=(
                bonus_step_limit(dist)
                if split_name == 'hard_bonus'
                else step_limit(dist)
            ),
        ))
        if progress_cb is not None:
            progress_cb()

    return episodes, attempts, max_attempts


def sample_episodes(
    graph: CSRGraph,
    conn: sqlite3.Connection,
    *,
    n_total: int = 100,
    n_hard_bonus: int = 3,
    seed: int = 42,
    max_sample_depth: int = 7,
    hard_source_max_out_degree: int = 20,
    hard_walk_max_out_degree: int = 50,
    episodes_path: Path = Path('episodes/benchmark.jsonl'),
    hard_bonus_path: Path | None = None,
    progress: bool = True,
) -> dict[str, list[Episode]]:
    """Sample a single reviewed benchmark set.

    Uses short random walks to propose candidates for each difficulty bucket,
    then verifies the exact shortest path with bidirectional BFS capped at
    ``max_sample_depth``. This is much cheaper than rejection-sampling random
    source/target pairs when we want rare 6-7 hop examples.
    """
    if progress:
        from tqdm import tqdm

    if max_sample_depth < 7 and n_hard_bonus > 0:
        raise ValueError('max_sample_depth must be at least 7 to support hard bonus episodes')

    idx_to_info: dict[int, tuple[int, str]] = {}
    active_indices: list[int] = []
    hard_source_pool: list[int] = []

    for row in conn.execute(
        "SELECT page_id, page_title, node_idx, out_degree FROM canonical ORDER BY node_idx"
    ):
        page_id, page_title, node_idx, out_degree = row
        idx_to_info[node_idx] = (page_id, page_title)
        if out_degree <= 0:
            continue
        active_indices.append(node_idx)
        if out_degree <= hard_source_max_out_degree:
            hard_source_pool.append(node_idx)

    if not active_indices:
        raise ValueError('no active canonical nodes available for sampling')
    if not hard_source_pool:
        hard_source_pool = active_indices

    targets = _default_bucket_targets(n_total)
    if hard_bonus_path is None:
        hard_bonus_path = episodes_path.with_name('hard_bonus.jsonl')
    print(
        f"Candidate pool: {len(active_indices):,} active, "
        f"{len(hard_source_pool):,} hard-source seeds (out_degree <= {hard_source_max_out_degree})"
    )
    print(
        f"Benchmark mix: easy={targets['easy']}, medium={targets['medium']}; "
        f"hard bonus={n_hard_bonus} (max depth {max_sample_depth})"
    )

    total_to_sample = n_total + n_hard_bonus
    pbar = tqdm(total=total_to_sample, desc='benchmark') if progress else None
    update_progress = pbar.update if pbar is not None else None
    rng = random.Random(seed)
    seen_pairs: set[tuple[int, int]] = set()
    benchmark_episodes: list[Episode] = []
    hard_bonus_episodes: list[Episode] = []

    bucket_specs = (
        ('medium', active_indices, (), {5}, 5, 'random_pair'),
        ('easy', active_indices, (4, 3, 4), {3, 4}, 4, 'walk'),
    )

    for (
        bucket_name,
        source_pool,
        walk_lengths,
        accepted_distances,
        verify_depth,
        proposal_mode,
    ) in bucket_specs:
        target_count = targets[bucket_name]
        bucket_episodes, attempts, max_attempts = _sample_bucket(
            graph,
            rng,
            idx_to_info,
            source_pool,
            split_name='benchmark',
            target_count=target_count,
            walk_lengths=walk_lengths,
            accepted_distances=accepted_distances,
            verify_depth=min(verify_depth, max_sample_depth),
            seen_pairs=seen_pairs,
            start_seed=len(benchmark_episodes),
            proposal_mode=proposal_mode,
            progress_cb=update_progress,
        )
        benchmark_episodes.extend(bucket_episodes)
        print(
            f"  {bucket_name}: sampled {len(bucket_episodes)}/{target_count} "
            f"in {attempts} attempts (cap {max_attempts})"
        )

    if n_hard_bonus > 0:
        bucket_episodes, attempts, max_attempts = _sample_bucket(
            graph,
            rng,
            idx_to_info,
            hard_source_pool,
            split_name='hard_bonus',
            target_count=n_hard_bonus,
            walk_lengths=(7, 6, 7),
            accepted_distances={6, 7},
            verify_depth=min(7, max_sample_depth),
            seen_pairs=seen_pairs,
            start_seed=0,
            max_neighbor_out_degree=hard_walk_max_out_degree,
            proposal_mode='walk',
            progress_cb=update_progress,
        )
        hard_bonus_episodes.extend(bucket_episodes)
        print(
            f"  hard_bonus: sampled {len(bucket_episodes)}/{n_hard_bonus} "
            f"in {attempts} attempts (cap {max_attempts})"
        )

    if pbar is not None:
        pbar.close()

    benchmark_episodes.sort(key=lambda episode: episode.seed)
    write_episodes_jsonl(benchmark_episodes, episodes_path)
    benchmark_csv_path = episodes_path.with_suffix('.csv')
    write_episodes_csv(benchmark_episodes, benchmark_csv_path)
    _write_dataset_metadata(
        episodes_path,
        benchmark_episodes,
        seed=seed,
        n_total=n_total,
        n_hard_bonus=n_hard_bonus,
        max_sample_depth=max_sample_depth,
        hard_source_max_out_degree=hard_source_max_out_degree,
        hard_walk_max_out_degree=hard_walk_max_out_degree,
    )
    print(f"Written {len(benchmark_episodes)} benchmark episodes to {episodes_path}")
    print(f"Written benchmark review CSV to {benchmark_csv_path}")
    print(f"Written benchmark metadata to {episodes_path.with_suffix('.meta.json')}")

    if hard_bonus_episodes:
        hard_bonus_episodes.sort(key=lambda episode: episode.seed)
        write_episodes_jsonl(hard_bonus_episodes, hard_bonus_path)
        hard_bonus_csv_path = hard_bonus_path.with_suffix('.csv')
        write_episodes_csv(hard_bonus_episodes, hard_bonus_csv_path)
        _write_dataset_metadata(
            hard_bonus_path,
            hard_bonus_episodes,
            seed=seed,
            n_total=n_total,
            n_hard_bonus=n_hard_bonus,
            max_sample_depth=max_sample_depth,
            hard_source_max_out_degree=hard_source_max_out_degree,
            hard_walk_max_out_degree=hard_walk_max_out_degree,
        )
        print(f"Written {len(hard_bonus_episodes)} hard bonus episodes to {hard_bonus_path}")
        print(f"Written hard bonus review CSV to {hard_bonus_csv_path}")
        print(f"Written hard bonus metadata to {hard_bonus_path.with_suffix('.meta.json')}")

    return {
        'benchmark': benchmark_episodes,
        'hard_bonus': hard_bonus_episodes,
    }
