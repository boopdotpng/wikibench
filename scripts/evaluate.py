"""Batch evaluation harness for the Wikipedia racing benchmark.

Runs a set of episodes through the benchmark engine, either:
1. With a simple built-in heuristic agent (for baseline/testing)
2. Via MCP server for external model evaluation

Produces per-episode traces and aggregate metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import click

from wikipedia_bench.db import open_db
from wikipedia_bench.graph import CSRGraph
from wikipedia_bench.mcp_server import WikiBenchEngine
from wikipedia_bench.schemas import load_episodes_jsonl


def _heuristic_pick(observation: dict, target_title: str) -> str | None:
    """Simple heuristic: pick the link whose title is closest to the target.

    Uses exact match first, then longest common substring as a tiebreaker.
    This is intentionally dumb — it's a baseline, not a real agent.
    """
    links = observation.get('links', [])
    if not links:
        return None

    target_lower = target_title.lower().replace('_', ' ')

    # Exact match
    for link in links:
        if link.replace('_', ' ').lower() == target_lower:
            return link

    # Score by word overlap with target
    target_words = set(target_lower.split())

    def _score(link: str) -> float:
        link_words = set(link.lower().replace('_', ' ').split())
        if not target_words:
            return 0
        overlap = len(target_words & link_words)
        return overlap / len(target_words)

    best = max(links, key=_score)
    if _score(best) > 0:
        return best

    # Fall back to first link (random-ish)
    return links[0]


def run_episode(engine: WikiBenchEngine, seed: int) -> dict:
    """Run a single episode with the heuristic agent. Returns trace dict."""
    result = engine.start_episode(seed)
    if 'error' in result:
        return {'seed': seed, 'error': result['error']}

    episode_id = result['episode_id']
    target_title = result['target_title']
    trace = {
        'seed': seed,
        'episode_id': episode_id,
        'start_title': result['start_title'],
        'target_title': target_title,
        'actions': [],
    }

    obs = result['observation']
    done = False
    step = 0

    while not done:
        pick = _heuristic_pick(obs, target_title)
        if pick is None:
            trace['actions'].append({'action': 'stuck', 'step': step})
            break

        t0 = time.time()
        click_result = engine.click_link(episode_id, pick)
        latency = time.time() - t0

        trace['actions'].append({
            'step': step,
            'clicked': pick,
            'ok': click_result.get('ok', False),
            'done': click_result.get('done', False),
            'error': click_result.get('error'),
            'latency_ms': round(latency * 1000, 1),
        })

        done = click_result.get('done', False)
        obs = click_result.get('observation', obs)
        step += 1

    # Score
    score = engine.score_ep(episode_id)
    trace['score'] = score
    return trace


def aggregate_metrics(traces: list[dict]) -> dict:
    """Compute aggregate metrics from a list of episode traces."""
    n_total = len(traces)
    n_errors = sum(1 for t in traces if 'error' in t and 'score' not in t)
    scored = [t for t in traces if 'score' in t]

    if not scored:
        return {
            'n_total': n_total,
            'n_errors': n_errors,
            'n_scored': 0,
        }

    successes = [t for t in scored if t['score'].get('success')]
    failures = [t for t in scored if not t['score'].get('success')]

    clicks_all = [t['score']['clicks'] for t in scored]
    clicks_success = [t['score']['clicks'] for t in successes]
    gaps = [t['score']['optimality_gap'] for t in successes
            if t['score'].get('optimality_gap') is not None]

    return {
        'n_total': n_total,
        'n_errors': n_errors,
        'n_scored': len(scored),
        'n_success': len(successes),
        'n_failure': len(failures),
        'success_rate': len(successes) / len(scored) if scored else 0,
        'avg_clicks_all': sum(clicks_all) / len(clicks_all) if clicks_all else 0,
        'avg_clicks_success': sum(clicks_success) / len(clicks_success) if clicks_success else 0,
        'avg_optimality_gap': sum(gaps) / len(gaps) if gaps else None,
        'median_clicks_success': sorted(clicks_success)[len(clicks_success) // 2] if clicks_success else None,
    }


@click.command()
@click.option('--db-path', default='data/processed/wiki.db',
              type=click.Path(exists=True))
@click.option('--graph-dir', default='data/processed',
              type=click.Path(exists=True))
@click.option('--episodes', default='episodes/dev.jsonl',
              type=click.Path(exists=True))
@click.option('--n', default=0, help='Number of episodes to run (0 = all)')
@click.option('--output', default='reports/eval_results.json',
              type=click.Path(), help='Output file for results')
def main(db_path: str, graph_dir: str, episodes: str, n: int, output: str) -> None:
    """Run batch evaluation with the built-in heuristic agent."""
    from tqdm import tqdm

    print("Loading engine...")
    engine = WikiBenchEngine(
        db_path=Path(db_path),
        graph_dir=Path(graph_dir),
        episodes_path=Path(episodes),
    )

    catalog = load_episodes_jsonl(Path(episodes))
    n_episodes = n if n > 0 else len(catalog)
    n_episodes = min(n_episodes, len(catalog))

    print(f"Running {n_episodes} episodes...")
    traces = []
    for seed in tqdm(range(n_episodes)):
        trace = run_episode(engine, seed)
        traces.append(trace)

    metrics = aggregate_metrics(traces)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        'metrics': metrics,
        'traces': traces,
    }, indent=2, default=str))

    print(f"\nResults written to {out_path}")
    print(f"\nAggregate metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
