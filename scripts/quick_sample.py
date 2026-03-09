from pathlib import Path
import random
import sys
import uuid

sys.path.insert(0, '.')

from wikipedia_bench.db import open_db
from wikipedia_bench.graph import CSRGraph, bidirectional_bfs
from wikipedia_bench.sampler import difficulty_bucket, step_limit
from wikipedia_bench.schemas import Episode, write_episodes_jsonl

DB_PATH = Path('data/processed/wiki.db')
GRAPH_DIR = Path('data/processed')
OUT_DIR = Path('episodes')
OUT_PATH = OUT_DIR / 'quick_benchmark.jsonl'


def main() -> None:
    print("Loading graph...")
    graph = CSRGraph.load(GRAPH_DIR)
    print(f"  {graph.n_nodes:,} nodes, {graph.n_fwd_edges:,} edges")

    conn = open_db(DB_PATH, readonly=True)
    idx_to_info = {
        row[2]: (row[0], row[1])
        for row in conn.execute("SELECT page_id, page_title, node_idx FROM canonical")
    }
    all_indices = list(idx_to_info)

    targets = {'easy': 10, 'medium': 10, 'hard': 2}
    counts = {bucket: 0 for bucket in targets}
    episodes: list[Episode] = []
    rng = random.Random(42)
    attempts = 0

    print("Sampling...")
    while any(counts[bucket] < target for bucket, target in targets.items()):
        attempts += 1
        if attempts > 5000:
            print(f"  Gave up after {attempts} attempts")
            break

        src = rng.choice(all_indices)
        dst = rng.choice(all_indices)
        if src == dst:
            continue

        max_depth = 7 if counts['hard'] < targets['hard'] else 5
        dist = bidirectional_bfs(graph, src, dst, max_depth=max_depth)
        if dist is None:
            continue

        bucket = difficulty_bucket(dist)
        if counts[bucket] >= targets[bucket]:
            continue

        src_pid, src_title = idx_to_info[src]
        dst_pid, dst_title = idx_to_info[dst]
        episodes.append(Episode(
            episode_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"wikibench.dev.{src_pid}.{dst_pid}.42").hex,
            split='dev',
            seed=len(episodes),
            start_page_id=src_pid,
            target_page_id=dst_pid,
            start_title=src_title,
            target_title=dst_title,
            shortest_path_len=dist,
            difficulty=bucket,
            step_limit=step_limit(dist),
        ))
        counts[bucket] += 1
        print(f"  [{len(episodes)}/22] {bucket} (d={dist}): {src_title} -> {dst_title}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_episodes_jsonl(episodes, OUT_PATH)
    print(f"\nDone: {len(episodes)} episodes in {attempts} attempts")
    for bucket in ['easy', 'medium', 'hard']:
        print(f"  {bucket}: {counts[bucket]}/{targets[bucket]}")
    print(f"Written to {OUT_PATH}")


if __name__ == '__main__':
    main()
