"""Integration test with synthetic data to validate the full pipeline.

Creates a small fake wiki graph and runs through:
1. DB creation + import
2. Redirect resolution
3. Graph construction
4. BFS
5. Episode sampling (mini)
6. Engine + scoring
"""

import gzip
import json
import sqlite3
import tempfile
from pathlib import Path

from wikipedia_bench.db import open_db, init_schema, bulk_insert, set_meta
from wikipedia_bench.redirects import build_canonical_table, build_redirect_map
from wikipedia_bench.graph import build_graph, bounded_bfs, bidirectional_bfs, CSRGraph
from wikipedia_bench.schemas import Episode, EpisodeState, PageObservation
from wikipedia_bench.scorer import score_episode


def _build_test_db(db_path: Path) -> sqlite3.Connection:
    """Create a test database with a small synthetic wiki graph.

    Graph structure:
        A -> B -> C -> D -> E
        A -> C (shortcut)
        B -> D (shortcut)
        F -> A (redirect page F points to A)
        G is a disambiguation page (excluded)
    """
    conn = open_db(db_path)
    init_schema(conn)

    # Insert pages (ns=0)
    pages = [
        (1, 0, 'A', 0, 100),
        (2, 0, 'B', 0, 200),
        (3, 0, 'C', 0, 150),
        (4, 0, 'D', 0, 300),
        (5, 0, 'E', 0, 250),
        (6, 0, 'F', 1, 50),   # redirect
        (7, 0, 'G', 0, 100),  # will be disambiguation
    ]
    bulk_insert(conn, 'page', ['page_id', 'page_ns', 'page_title', 'is_redirect', 'page_len'], pages)

    # Insert redirect: F -> A
    bulk_insert(conn, 'redirect_raw', ['rd_from', 'rd_ns', 'rd_title'], [
        (6, 0, 'A'),
    ])

    # Insert page_props: G is disambiguation
    bulk_insert(conn, 'page_props', ['pp_page', 'pp_propname', 'pp_value'], [
        (7, 'disambiguation', ''),
    ])

    # Insert linktargets
    linktargets = [
        (101, 0, 'B'),
        (102, 0, 'C'),
        (103, 0, 'D'),
        (104, 0, 'E'),
        (105, 0, 'A'),
        (106, 0, 'F'),  # redirect target
        (107, 0, 'G'),  # disambiguation target
    ]
    bulk_insert(conn, 'linktarget', ['lt_id', 'lt_ns', 'lt_title'], linktargets)

    # Insert pagelinks (pl_from, pl_target_id)
    # A -> B, A -> C
    # B -> C, B -> D
    # C -> D
    # D -> E
    # Also: A -> F (should resolve to A = self-loop, dropped)
    # Also: A -> G (disambiguation, should be dropped since G is excluded)
    pagelinks = [
        (1, 101),  # A -> B
        (1, 102),  # A -> C
        (2, 102),  # B -> C
        (2, 103),  # B -> D
        (3, 103),  # C -> D
        (4, 104),  # D -> E
        (1, 106),  # A -> F (redirect to A, self-loop)
        (1, 107),  # A -> G (disambiguation, excluded)
    ]
    bulk_insert(conn, 'pagelinks_raw', ['pl_from', 'pl_target_id'], pagelinks)

    return conn


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        db_path = tmpdir / 'test.db'
        graph_dir = tmpdir / 'graph'

        # 1. Build DB
        conn = _build_test_db(db_path)

        # 2. Build canonical table
        n_canonical = build_canonical_table(conn)
        assert n_canonical == 5, f"Expected 5 canonical pages (A-E), got {n_canonical}"

        # Verify G (disambiguation) is excluded
        row = conn.execute("SELECT COUNT(*) FROM canonical WHERE page_title = 'G'").fetchone()
        assert row[0] == 0, "Disambiguation page G should be excluded"

        # Verify F (redirect) is excluded
        row = conn.execute("SELECT COUNT(*) FROM canonical WHERE page_title = 'F'").fetchone()
        assert row[0] == 0, "Redirect page F should be excluded"

        # 3. Build redirect map
        n_redirects = build_redirect_map(conn)
        assert n_redirects >= 1, f"Expected at least 1 redirect, got {n_redirects}"

        # Verify F -> A resolution
        row = conn.execute(
            "SELECT canonical_id FROM redirect_map WHERE rd_from_id = 6"
        ).fetchone()
        assert row is not None and row[0] == 1, "Redirect F(6) should resolve to A(1)"

        # 4. Build graph
        graph = build_graph(conn, graph_dir, progress=False)
        assert graph.n_nodes == 5, f"Expected 5 nodes, got {graph.n_nodes}"

        # Build lookup
        pid_to_idx = {}
        idx_to_title = {}
        for row in conn.execute("SELECT page_id, page_title, node_idx FROM canonical"):
            pid_to_idx[row[0]] = row[2]
            idx_to_title[row[2]] = row[1]

        # Verify edges: A->B, A->C, B->C, B->D, C->D, D->E = 6 edges
        # A->F (resolves to A, self-loop) should be dropped
        # A->G (disambiguation) should be dropped
        assert graph.n_fwd_edges == 6, f"Expected 6 edges, got {graph.n_fwd_edges}"

        # 5. Test BFS
        a_idx = pid_to_idx[1]  # A
        e_idx = pid_to_idx[5]  # E

        # A -> E shortest path: A->C->D->E = 3, or A->B->D->E = 3
        dist = bidirectional_bfs(graph, a_idx, e_idx)
        assert dist == 3, f"Expected shortest path A->E = 3, got {dist}"

        # A -> B = 1
        b_idx = pid_to_idx[2]
        dist = bidirectional_bfs(graph, a_idx, b_idx)
        assert dist == 1, f"Expected shortest path A->B = 1, got {dist}"

        # Bounded BFS from A, depth 2
        reachable = bounded_bfs(graph, a_idx, 2)
        assert a_idx in reachable and reachable[a_idx] == 0
        assert b_idx in reachable and reachable[b_idx] == 1
        c_idx = pid_to_idx[3]
        d_idx = pid_to_idx[4]
        assert c_idx in reachable and reachable[c_idx] == 1  # A->C
        assert d_idx in reachable and reachable[d_idx] == 2  # A->C->D or A->B->D

        # 6. Test graph save/load roundtrip
        graph2 = CSRGraph.load(graph_dir)
        assert graph2.n_nodes == graph.n_nodes
        assert graph2.n_fwd_edges == graph.n_fwd_edges

        # 7. Test scoring
        ep = Episode(
            episode_id='test-ep',
            split='test',
            seed=0,
            start_page_id=1,
            target_page_id=5,
            start_title='A',
            target_title='E',
            shortest_path_len=3,
            difficulty='easy',
            step_limit=12,
        )
        state = EpisodeState(
            episode=ep,
            current_page_id=5,  # reached target
            path=[1, 3, 4, 5],  # A -> C -> D -> E
            clicks=3,
            invalid_actions=0,
            terminated=True,
            terminated_reason='success',
        )
        result = score_episode(state, graph, idx_to_title, pid_to_idx)
        assert result.success is True
        assert result.clicks == 3
        assert result.shortest_path_len == 3
        assert result.optimality_gap == 0
        assert result.path == ['A', 'C', 'D', 'E']

        conn.close()
        print("All integration tests passed!")


if __name__ == '__main__':
    test_full_pipeline()
