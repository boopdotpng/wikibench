"""Episode scoring utilities."""

from __future__ import annotations

from .graph import CSRGraph, bidirectional_bfs
from .schemas import EpisodeState, ScoreResult


def score_episode(
    state: EpisodeState,
    graph: CSRGraph,
    idx_to_title: dict[int, str],
    lookup_pid_to_idx,
) -> ScoreResult:
    """Compute final score for a completed episode.

    Args:
        lookup_pid_to_idx: callable(pid) -> idx or None, OR dict with .get()
    """
    # Support both callable and dict interfaces
    _pid_to_idx = (
        lookup_pid_to_idx if callable(lookup_pid_to_idx)
        else lookup_pid_to_idx.get
    )

    # Build path of titles
    path_titles = [
        idx_to_title.get(idx, f"?{pid}") if (idx := _pid_to_idx(pid)) is not None else f"?{pid}"
        for pid in state.path
    ]

    # Shortest path (recompute via bidi-BFS for accuracy)
    src_idx = _pid_to_idx(state.episode.start_page_id)
    dst_idx = _pid_to_idx(state.episode.target_page_id)
    if src_idx is not None and dst_idx is not None:
        spl = bidirectional_bfs(graph, src_idx, dst_idx)
    else:
        spl = None

    # Use pre-sampled shortest_path_len as fallback
    if spl is None:
        spl = state.episode.shortest_path_len

    success = (
        state.terminated_reason == 'success'
        or state.current_page_id == state.episode.target_page_id
    )

    gap = (state.clicks - spl) if spl is not None and success else None

    return ScoreResult(
        success=success,
        clicks=state.clicks,
        path=path_titles,
        shortest_path_len=spl,
        optimality_gap=gap,
        invalid_actions=state.invalid_actions,
        terminated_reason=state.terminated_reason or 'unknown',
    )
