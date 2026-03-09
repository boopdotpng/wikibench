"""Quick episode sampler — generates exactly 10 easy + 10 medium + 2 hard."""

import sys
import random
import sqlite3
import uuid
sys.path.insert(0, '.')

from wikipedia_bench.graph import CSRGraph, bidirectional_bfs
from wikipedia_bench.schemas import Episode, write_episodes_jsonl
from wikipedia_bench.db import open_db
from pathlib import Path

DB_PATH = Path('data/processed/wiki.db')
GRAPH_DIR = Path('data/processed')
OUT_DIR = Path('episodes')

def step_limit(d):
    return max(2 * d + 4, 12)

def difficulty_bucket(d):
    if d <= 3: return 'easy'
    if d <= 5: return 'medium'
    return 'hard'

print("Loading graph...")
graph = CSRGraph.load(GRAPH_DIR)
print(f"  {graph.n_nodes:,} nodes, {graph.n_fwd_edges:,} edges")

conn = open_db(DB_PATH, readonly=True)
idx_to_info = {}
for row in conn.execute("SELECT page_id, page_title, node_idx FROM canonical"):
    idx_to_info[row[2]] = (row[0], row[1])
all_indices = list(idx_to_info.keys())

targets = {'easy': 10, 'medium': 10, 'hard': 2}
counts = {'easy': 0, 'medium': 0, 'hard': 0}
episodes = []
rng = random.Random(42)
attempts = 0

print("Sampling...")
while any(counts[b] < targets[b] for b in targets):
    attempts += 1
    if attempts > 5000:
        print(f"  Gave up after {attempts} attempts")
        break

    src = rng.choice(all_indices)
    dst = rng.choice(all_indices)
    if src == dst:
        continue

    # For hard, allow depth 6; for easy/medium, depth 3 is enough
    need_hard = counts['hard'] < targets['hard']
    max_d = 6 if need_hard else 3

    dist = bidirectional_bfs(graph, src, dst, max_depth=max_d)
    if dist is None:
        continue

    bucket = difficulty_bucket(dist)
    if counts[bucket] >= targets[bucket]:
        continue

    src_pid, src_title = idx_to_info[src]
    dst_pid, dst_title = idx_to_info[dst]

    ep = Episode(
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
    )
    episodes.append(ep)
    counts[bucket] += 1
    print(f"  [{len(episodes)}/22] {bucket} (d={dist}): {src_title} -> {dst_title}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
write_episodes_jsonl(episodes, OUT_DIR / 'dev.jsonl')
print(f"\nDone: {len(episodes)} episodes in {attempts} attempts")
for b in ['easy', 'medium', 'hard']:
    print(f"  {b}: {counts[b]}/{targets[b]}")
print(f"Written to {OUT_DIR / 'dev.jsonl'}")
