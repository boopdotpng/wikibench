"""CSR directed graph with BFS utilities for the Wikipedia benchmark."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections import deque
from pathlib import Path

import numpy as np

# Binary file header: uint64 n_nodes, uint64 n_edges
_HEADER_FMT = '<QQ'
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class CSRGraph:
    """Compressed Sparse Row directed graph (forward + backward).

    Nodes are identified by contiguous 0-based indices (node_idx).
    """

    def __init__(
        self,
        fwd_indptr: np.ndarray,
        fwd_indices: np.ndarray,
        bwd_indptr: np.ndarray,
        bwd_indices: np.ndarray,
        n_nodes: int,
        n_fwd_edges: int,
        n_bwd_edges: int,
    ):
        self.fwd_indptr = fwd_indptr
        self.fwd_indices = fwd_indices
        self.bwd_indptr = bwd_indptr
        self.bwd_indices = bwd_indices
        self.n_nodes = n_nodes
        self.n_fwd_edges = n_fwd_edges
        self.n_bwd_edges = n_bwd_edges

    def neighbors(self, node_idx: int) -> np.ndarray:
        """Forward neighbors of node_idx."""
        start = self.fwd_indptr[node_idx]
        end = self.fwd_indptr[node_idx + 1]
        return self.fwd_indices[start:end]

    def in_neighbors(self, node_idx: int) -> np.ndarray:
        """Backward (predecessor) neighbors of node_idx."""
        start = self.bwd_indptr[node_idx]
        end = self.bwd_indptr[node_idx + 1]
        return self.bwd_indices[start:end]

    def out_degree(self, node_idx: int) -> int:
        return int(self.fwd_indptr[node_idx + 1] - self.fwd_indptr[node_idx])

    def in_degree(self, node_idx: int) -> int:
        return int(self.bwd_indptr[node_idx + 1] - self.bwd_indptr[node_idx])

    def save(self, out_dir: Path) -> None:
        """Save graph arrays to binary files + metadata JSON."""
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_array(out_dir / 'graph.fwd.indptr', self.fwd_indptr, self.n_nodes, self.n_fwd_edges)
        _save_array(out_dir / 'graph.fwd.indices', self.fwd_indices, self.n_nodes, self.n_fwd_edges)
        _save_array(out_dir / 'graph.bwd.indptr', self.bwd_indptr, self.n_nodes, self.n_bwd_edges)
        _save_array(out_dir / 'graph.bwd.indices', self.bwd_indices, self.n_nodes, self.n_bwd_edges)
        meta = {
            'n_nodes': self.n_nodes,
            'n_fwd_edges': self.n_fwd_edges,
            'n_bwd_edges': self.n_bwd_edges,
            'dtype': 'int32',
        }
        (out_dir / 'graph_meta.json').write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, graph_dir: Path) -> CSRGraph:
        """Load graph from binary files."""
        meta = json.loads((graph_dir / 'graph_meta.json').read_text())
        n_nodes = meta['n_nodes']
        n_fwd = meta['n_fwd_edges']
        n_bwd = meta['n_bwd_edges']
        return cls(
            fwd_indptr=_load_array(graph_dir / 'graph.fwd.indptr'),
            fwd_indices=_load_array(graph_dir / 'graph.fwd.indices'),
            bwd_indptr=_load_array(graph_dir / 'graph.bwd.indptr'),
            bwd_indices=_load_array(graph_dir / 'graph.bwd.indices'),
            n_nodes=n_nodes,
            n_fwd_edges=n_fwd,
            n_bwd_edges=n_bwd,
        )


def _save_array(path: Path, arr: np.ndarray, n_nodes: int, n_edges: int) -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack(_HEADER_FMT, n_nodes, n_edges))
        arr.tofile(f)


def _load_array(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=np.int32, mode='r', offset=_HEADER_SIZE)


def build_graph(
    conn: sqlite3.Connection,
    out_dir: Path,
    *,
    progress: bool = True,
) -> CSRGraph:
    """Build CSR graph from wiki.db and save to out_dir.

    Steps:
    1. Load canonical page_id -> node_idx mapping
    2. Load redirect_map for target resolution
    3. Stream pagelinks joined with linktarget (ns0 only)
    4. Resolve targets through redirects to canonical node_idx
    5. Build sorted deduplicated edge list
    6. Construct forward + backward CSR arrays
    """
    if progress:
        from tqdm import tqdm

    # 1. canonical page_id -> node_idx
    print("Loading canonical mapping...")
    pid_to_idx: dict[int, int] = {}
    for row in conn.execute("SELECT page_id, node_idx FROM canonical"):
        pid_to_idx[row[0]] = row[1]
    n_nodes = len(pid_to_idx)
    print(f"  {n_nodes:,} canonical nodes")

    # 2. redirect_map
    print("Loading redirect map...")
    redirect: dict[int, int | None] = {}
    for row in conn.execute("SELECT rd_from_id, canonical_id FROM redirect_map"):
        redirect[row[0]] = row[1]
    print(f"  {len(redirect):,} redirects")

    # 3. title -> page_id for ns0 (needed to resolve linktarget titles)
    print("Loading title -> page_id...")
    title_to_pid: dict[str, int] = {}
    for row in conn.execute("SELECT page_id, page_title FROM page WHERE page_ns = 0"):
        title_to_pid[row[1]] = row[0]
    print(f"  {len(title_to_pid):,} ns0 titles")

    # 4. Stream edges
    print("Streaming edges from pagelinks + linktarget...")
    query = """\
        SELECT pl.pl_from, lt.lt_title
        FROM pagelinks_raw pl
        JOIN linktarget lt ON pl.pl_target_id = lt.lt_id
        WHERE lt.lt_ns = 0
    """

    # Stream edges to a temp binary file to avoid Python list memory overhead.
    # A Python list of ints uses ~28 bytes/int; a temp file uses 4 bytes/int32.
    import struct as _struct
    import tempfile as _tempfile

    out_dir.mkdir(parents=True, exist_ok=True)
    edge_tmp = _tempfile.NamedTemporaryFile(suffix='.edges.bin', dir=str(out_dir), delete=True)
    _pack = _struct.Struct('<ii').pack
    n_raw = 0
    n_kept = 0

    import time as _time
    cursor = conn.execute(query)
    batch_size = 500_000
    _t0 = _time.time()
    _last_report = _t0

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            n_raw += 1
            pl_from = row[0]
            lt_title = row[1]

            # Source must be canonical
            src_idx = pid_to_idx.get(pl_from)
            if src_idx is None:
                continue

            # Resolve target title -> page_id
            dst_pid = title_to_pid.get(lt_title)
            if dst_pid is None:
                continue

            # If target is a redirect, follow it
            if dst_pid in redirect:
                canonical_pid = redirect[dst_pid]
                if canonical_pid is None:
                    continue
                dst_pid = canonical_pid

            # Target must be canonical
            dst_idx = pid_to_idx.get(dst_pid)
            if dst_idx is None:
                continue

            # Skip self-loops
            if src_idx == dst_idx:
                continue

            edge_tmp.write(_pack(src_idx, dst_idx))
            n_kept += 1

        _now = _time.time()
        if progress and _now - _last_report >= 5.0:
            _elapsed = _now - _t0
            _rate = n_raw / _elapsed if _elapsed > 0 else 0
            print(f"  {n_raw:,} raw edges, {n_kept:,} kept ({_rate:,.0f} edges/s)")
            _last_report = _now

    print(f"  total: {n_raw:,} raw, {n_kept:,} kept (before dedup)")

    # Free lookup dicts before memory-heavy numpy operations
    del pid_to_idx, redirect, title_to_pid
    import gc; gc.collect()

    # 5. Load edges from temp file into numpy, sort, dedup
    print("Sorting and deduplicating edges...")
    if n_kept == 0:
        edges = np.empty((0, 2), dtype=np.int32)
        edge_tmp.close()
    else:
        edge_tmp.seek(0)
        edges = np.fromfile(edge_tmp, dtype=np.int32).reshape(-1, 2)
        edge_tmp.close()  # deletes temp file

        # Sort by (src, dst) for CSR construction and dedup
        # Use a combined key to avoid lexsort's extra memory (lexsort needs 8 bytes/elem)
        # Combine src (int32) and dst (int32) into a single int64 for sorting
        print("  Computing sort keys...")
        sort_key = edges[:, 0].astype(np.int64) * (np.int64(n_nodes) + 1) + edges[:, 1].astype(np.int64)
        print("  Sorting...")
        order = np.argsort(sort_key, kind='mergesort')
        del sort_key
        edges = edges[order]
        del order

        # Remove duplicates
        if len(edges) > 1:
            mask = np.ones(len(edges), dtype=bool)
            mask[1:] = (edges[1:, 0] != edges[:-1, 0]) | (edges[1:, 1] != edges[:-1, 1])
            edges = edges[mask]
            del mask

    n_edges = len(edges)
    print(f"  {n_edges:,} unique forward edges")

    # 6. Build forward CSR
    print("Building forward CSR...")
    fwd_indptr, fwd_indices = _build_csr(edges, n_nodes)

    # 7. Build backward CSR (transpose)
    # Reuse edges array columns swapped, sort in-place to minimize copies
    print("Building backward CSR...")
    if n_edges > 0:
        # Swap columns in-place
        edges[:, 0], edges[:, 1] = edges[:, 1].copy(), edges[:, 0].copy()
        sort_key = edges[:, 0].astype(np.int64) * (np.int64(n_nodes) + 1) + edges[:, 1].astype(np.int64)
        order = np.argsort(sort_key, kind='mergesort')
        del sort_key
        edges = edges[order]
        del order
    bwd_indptr, bwd_indices = _build_csr(edges, n_nodes)
    n_bwd = len(bwd_indices)
    del edges

    # Update out_degree in canonical table
    print("Updating out_degree in canonical table...")
    degrees = np.diff(fwd_indptr).astype(np.int32, copy=False)
    # Reload the mapping from DB (we freed the dict earlier to save memory)
    page_ids_by_idx = np.empty(n_nodes, dtype=np.int64)
    for row in conn.execute("SELECT page_id, node_idx FROM canonical"):
        page_ids_by_idx[row[1]] = row[0]
    batch = list(zip(degrees.tolist(), page_ids_by_idx.tolist()))
    conn.executemany("UPDATE canonical SET out_degree = ? WHERE page_id = ?", batch)
    conn.commit()
    del batch, page_ids_by_idx

    graph = CSRGraph(
        fwd_indptr=fwd_indptr,
        fwd_indices=fwd_indices,
        bwd_indptr=bwd_indptr,
        bwd_indices=bwd_indices,
        n_nodes=n_nodes,
        n_fwd_edges=n_edges,
        n_bwd_edges=n_bwd,
    )

    print("Saving graph...")
    graph.save(out_dir)
    print("Done.")
    return graph


def _build_csr(
    sorted_edges: np.ndarray,  # shape (n_edges, 2), sorted by col 0 then col 1
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR arrays from sorted (src, dst) edge pairs."""
    n_edges = len(sorted_edges)
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    indices = np.empty(n_edges, dtype=np.int32)

    if n_edges > 0:
        # Count edges per source
        np.add.at(indptr[1:], sorted_edges[:, 0], 1)
        np.cumsum(indptr, out=indptr)
        indices[:] = sorted_edges[:, 1]

    return indptr, indices


# --- BFS utilities ---


def bounded_bfs(
    graph: CSRGraph,
    src_idx: int,
    max_depth: int,
    *,
    direction: str = 'fwd',
) -> dict[int, int]:
    """BFS from src_idx up to max_depth hops.

    Returns {node_idx: distance} for all reachable nodes within max_depth.
    """
    get_neighbors = graph.neighbors if direction == 'fwd' else graph.in_neighbors
    dist: dict[int, int] = {src_idx: 0}
    frontier = deque([src_idx])

    while frontier:
        node = frontier.popleft()
        d = dist[node]
        if d >= max_depth:
            continue
        for nb in get_neighbors(node):
            nb = int(nb)
            if nb not in dist:
                dist[nb] = d + 1
                frontier.append(nb)

    return dist


def bidirectional_bfs(
    graph: CSRGraph,
    src_idx: int,
    dst_idx: int,
    max_depth: int = 30,
) -> int | None:
    """Find shortest path length between src and dst using bidirectional BFS.

    Returns None if not reachable within max_depth.
    """
    if src_idx == dst_idx:
        return 0

    # Forward BFS state
    fwd_dist: dict[int, int] = {src_idx: 0}
    fwd_frontier: list[int] = [src_idx]
    fwd_depth = 0

    # Backward BFS state (traverse incoming edges from dst)
    bwd_dist: dict[int, int] = {dst_idx: 0}
    bwd_frontier: list[int] = [dst_idx]
    bwd_depth = 0

    best = max_depth + 1  # infinity sentinel

    while fwd_frontier or bwd_frontier:
        # Stop if both depths exceed the best found
        if fwd_depth + bwd_depth >= best:
            break
        if fwd_depth > max_depth and bwd_depth > max_depth:
            break

        # Expand the smaller frontier
        expand_fwd = True
        if not fwd_frontier:
            expand_fwd = False
        elif bwd_frontier and len(bwd_frontier) < len(fwd_frontier):
            expand_fwd = False

        if expand_fwd and fwd_depth < max_depth:
            fwd_depth += 1
            next_frontier: list[int] = []
            for node in fwd_frontier:
                for nb in graph.neighbors(node):
                    nb = int(nb)
                    if nb not in fwd_dist:
                        fwd_dist[nb] = fwd_depth
                        next_frontier.append(nb)
                        if nb in bwd_dist:
                            cand = fwd_depth + bwd_dist[nb]
                            if cand < best:
                                best = cand
            fwd_frontier = next_frontier
        elif bwd_frontier and bwd_depth < max_depth:
            bwd_depth += 1
            next_frontier = []
            for node in bwd_frontier:
                for nb in graph.in_neighbors(node):
                    nb = int(nb)
                    if nb not in bwd_dist:
                        bwd_dist[nb] = bwd_depth
                        next_frontier.append(nb)
                        if nb in fwd_dist:
                            cand = bwd_depth + fwd_dist[nb]
                            if cand < best:
                                best = cand
            bwd_frontier = next_frontier
        else:
            break

    return best if best <= max_depth else None
