"""MCP server for the Wikipedia racing benchmark.

Exposes the benchmark as MCP tools: start_episode, get_page, click_link,
search_page, score_episode.
"""

from __future__ import annotations

import json
import pickle
import re
import sqlite3
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .article_reader import ArticleReader, ParsedArticle, parse_article_sections
from .graph import CSRGraph
from .schemas import (
    EpisodeState, PageObservation, SectionSummary,
    load_episodes_jsonl,
)
from .scorer import score_episode
from .wikitext import normalize_title

# Episode idle expiry in seconds
EPISODE_EXPIRY = 30 * 60  # 30 minutes
INVALID_ACTION_LIMIT = 3


class WikiBenchEngine:
    """Core benchmark engine managing episodes and game state."""

    def __init__(
        self,
        db_path: Path,
        graph_dir: Path,
        episodes_path: Path,
        dump_path: Path | None = None,
        index_path: Path | None = None,
        snapshot: str = "enwiki-latest",
    ):
        from .db import open_db

        self.conn = open_db(db_path, readonly=True)
        self.graph = CSRGraph.load(graph_dir)
        self.episodes_catalog = load_episodes_jsonl(episodes_path)
        self.snapshot = snapshot

        # Article reader for full article text (on-the-fly from XML dump)
        self._article_reader: ArticleReader | None = None
        if dump_path and index_path:
            self._article_reader = ArticleReader(dump_path, index_path)
            self._article_reader.load_index()

        # Build lookup tables (with binary cache for fast restart)
        self._load_lookups(graph_dir)

        # Active episodes
        self._episodes: dict[str, EpisodeState] = {}

        # Cache parsed article per episode (for search_page and get_section)
        self._last_parsed: dict[str, ParsedArticle] = {}
        # Cache legal link set per episode (so get_section can filter)
        self._last_legal_links: dict[str, set[str]] = {}

    def _load_lookups(self, graph_dir: Path) -> None:
        """Load pid/idx/title/redirect lookup tables, using a binary cache."""
        cache_dir = graph_dir / '.lookup_cache'
        db_mtime = Path(self.conn.execute(
            "PRAGMA database_list"
        ).fetchone()[2]).stat().st_mtime

        # Check if cache is fresh
        meta_path = cache_dir / 'meta.json'
        if cache_dir.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get('db_mtime') == db_mtime:
                    self._load_lookups_from_cache(cache_dir)
                    return
            except Exception:
                pass  # Fall through to rebuild

        # Build from DB and save cache
        self._build_lookups_from_db()
        try:
            self._save_lookup_cache(cache_dir, db_mtime)
        except Exception:
            pass  # Non-fatal

    def _load_lookups_from_cache(self, cache_dir: Path) -> None:
        """Load lookup tables from mmap'd binary cache."""
        # Direct-index arrays: arr[page_id] = node_idx (or -1)
        self._pid_to_idx_arr = np.memmap(
            cache_dir / 'pid_to_idx.bin', dtype=np.int32, mode='r')
        # Direct-index array: arr[node_idx] = page_id
        self._idx_to_pid_arr = np.memmap(
            cache_dir / 'idx_to_pid.bin', dtype=np.int32, mode='r')
        # Redirect: arr[rd_from_id] = canonical_id (or -1)
        self._redirect_arr = np.memmap(
            cache_dir / 'redirect.bin', dtype=np.int32, mode='r')
        # idx_to_title: flat list indexed by node_idx
        with open(cache_dir / 'idx_to_title.pkl', 'rb') as f:
            title_list: list[str] = pickle.load(f)
        self._idx_to_title = {i: t for i, t in enumerate(title_list) if t}

    def _build_lookups_from_db(self) -> None:
        """Build lookup tables by scanning DB (slow path, first run only)."""
        self._idx_to_title = {}

        # First pass: find max page_id and n_nodes
        row = self.conn.execute(
            "SELECT MAX(page_id), MAX(node_idx) FROM canonical"
        ).fetchone()
        max_pid, max_idx = row[0], row[1]

        self._pid_to_idx_arr = np.full(max_pid + 1, -1, dtype=np.int32)
        self._idx_to_pid_arr = np.empty(max_idx + 1, dtype=np.int32)

        for r in self.conn.execute("SELECT page_id, page_title, node_idx FROM canonical"):
            pid, title, idx = r[0], r[1], r[2]
            self._pid_to_idx_arr[pid] = idx
            self._idx_to_pid_arr[idx] = pid
            self._idx_to_title[idx] = title

        # Redirect map
        row = self.conn.execute("SELECT MAX(rd_from_id) FROM redirect_map").fetchone()
        max_rd = row[0] if row[0] is not None else 0
        self._redirect_arr = np.full(max_rd + 1, -1, dtype=np.int32)
        for r in self.conn.execute("SELECT rd_from_id, canonical_id FROM redirect_map"):
            if r[1] is not None:
                self._redirect_arr[r[0]] = r[1]

    def _save_lookup_cache(self, cache_dir: Path, db_mtime: float) -> None:
        """Save lookup tables as binary cache files."""
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        self._pid_to_idx_arr.tofile(cache_dir / 'pid_to_idx.bin')
        self._idx_to_pid_arr.tofile(cache_dir / 'idx_to_pid.bin')
        self._redirect_arr.tofile(cache_dir / 'redirect.bin')

        # Save idx_to_title as a flat list (indexed by node_idx)
        max_idx = max(self._idx_to_title.keys()) if self._idx_to_title else 0
        title_list = [''] * (max_idx + 1)
        for idx, title in self._idx_to_title.items():
            title_list[idx] = title
        with open(cache_dir / 'idx_to_title.pkl', 'wb') as f:
            pickle.dump(title_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata
        meta = {'db_mtime': db_mtime}
        (cache_dir / 'meta.json').write_text(json.dumps(meta))

    def start_episode(self, seed: int | None = None) -> dict:
        """Start a new episode. Seed indexes into the pre-sampled catalog."""
        self._expire_idle()

        if seed is not None:
            if seed < 0 or seed >= len(self.episodes_catalog):
                return {"error": f"seed {seed} out of range [0, {len(self.episodes_catalog)})"}
            ep = self.episodes_catalog[seed]
            episode_id = ep.episode_id
        else:
            ep = self.episodes_catalog[0]
            episode_id = uuid.uuid4().hex

        now = time.time()
        state = EpisodeState(
            episode=ep,
            current_page_id=ep.start_page_id,
            path=[ep.start_page_id],
            clicks=0,
            invalid_actions=0,
            terminated=False,
            terminated_reason=None,
            created_at=now,
            last_active_at=now,
        )
        self._episodes[episode_id] = state

        obs = self._make_observation(state, episode_id)
        return {
            "episode_id": episode_id,
            "snapshot": self.snapshot,
            "start_title": ep.start_title,
            "target_title": ep.target_title,
            "step_limit": ep.step_limit,
            "observation": asdict(obs),
        }

    def get_page(self, episode_id: str) -> dict:
        """Return current page observation."""
        state = self._get_state(episode_id)
        if state is None:
            return {"error": f"episode {episode_id} not found"}
        state.last_active_at = time.time()
        obs = self._make_observation(state, episode_id)
        return asdict(obs)

    def click_link(self, episode_id: str, title: str) -> dict:
        """Attempt to click a link on the current page."""
        state = self._get_state(episode_id)
        if state is None:
            return {"ok": False, "error": f"episode {episode_id} not found", "done": False}

        if state.terminated:
            return {
                "ok": False,
                "done": True,
                "error": f"episode already terminated: {state.terminated_reason}",
                "observation": asdict(self._make_observation(state, episode_id)),
            }

        state.last_active_at = time.time()

        # Normalize the clicked title
        title_norm = normalize_title(title)

        # Check legality: title must be in the current page's link set
        current_links = self._get_page_links(state.current_page_id)
        if title_norm not in current_links:
            return self._invalid_click(
                state,
                episode_id,
                f"'{title}' is not a valid link on the current page",
            )

        # Resolve the target (follows redirects to canonical page)
        dst_pid = self._resolve_title_to_pid(title_norm)
        if dst_pid is None:
            return self._invalid_click(
                state,
                episode_id,
                f"'{title}' could not be resolved to a canonical page",
            )

        # Move
        state.clicks += 1
        state.current_page_id = dst_pid
        state.path.append(dst_pid)

        # Check success
        done = False
        if dst_pid == state.episode.target_page_id:
            state.terminated = True
            state.terminated_reason = 'success'
            done = True
        elif state.clicks >= state.episode.step_limit:
            state.terminated = True
            state.terminated_reason = 'step_limit'
            done = True

        obs = self._make_observation(state, episode_id)
        return {
            "ok": True,
            "done": done,
            "observation": asdict(obs),
        }

    def get_section(self, episode_id: str, section: str) -> dict:
        """Return the full text of a specific section."""
        state = self._get_state(episode_id)
        if state is None:
            return {"error": f"episode {episode_id} not found"}

        parsed = self._last_parsed.get(episode_id)
        if parsed is None:
            return {"error": "no article loaded for this episode"}

        legal_links = self._last_legal_links.get(episode_id, set())

        for sec in parsed.sections:
            if sec.name.lower() == section.lower():
                return {
                    "section": sec.name,
                    "text": sec.text,
                    "links": [{"title": lc.title, "context": lc.context}
                              for lc in sec.links if lc.title in legal_links],
                }

        available = [s.name for s in parsed.sections]
        return {"error": f"section '{section}' not found", "available_sections": available}

    def search_page(self, episode_id: str, query: str, max_results: int = 10) -> dict:
        """Search the current article text for a query string (like ctrl+F)."""
        state = self._get_state(episode_id)
        if state is None:
            return {"error": f"episode {episode_id} not found"}

        parsed = self._last_parsed.get(episode_id)
        article = parsed.full_text if parsed else ''
        if not article:
            return {"error": "no article text loaded for this episode", "matches": []}

        # Case-insensitive search, return matching lines with context
        matches = []
        lines = article.split('\n')
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for i, line in enumerate(lines):
            if pattern.search(line):
                # Include 1 line of context before/after
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = '\n'.join(lines[start:end])
                matches.append({
                    "line_number": i + 1,
                    "context": context,
                })
                if len(matches) >= max_results:
                    break

        return {
            "query": query,
            "total_matches": sum(1 for line in lines if pattern.search(line)),
            "matches": matches,
        }

    def score_ep(self, episode_id: str) -> dict:
        """Score a completed (or in-progress) episode."""
        state = self._get_state(episode_id)
        if state is None:
            return {"error": f"episode {episode_id} not found"}

        result = score_episode(state, self.graph, self._idx_to_title, self._lookup_pid_to_idx)
        return asdict(result)

    def _get_state(self, episode_id: str) -> EpisodeState | None:
        return self._episodes.get(episode_id)

    def _invalid_click(self, state: EpisodeState, episode_id: str, error: str) -> dict:
        state.invalid_actions += 1
        if state.invalid_actions >= INVALID_ACTION_LIMIT:
            state.terminated = True
            state.terminated_reason = 'invalid_action_limit'
        return {
            "ok": False,
            "done": state.terminated,
            "error": error,
            "observation": asdict(self._make_observation(state, episode_id)),
        }

    def _lookup_pid_to_idx(self, pid: int) -> int | None:
        """Look up node_idx for a page_id. Returns None if not found."""
        if pid < 0 or pid >= len(self._pid_to_idx_arr):
            return None
        val = int(self._pid_to_idx_arr[pid])
        return val if val >= 0 else None

    def _lookup_idx_to_title(self, idx: int) -> str | None:
        """Look up title for a node_idx."""
        return self._idx_to_title.get(idx)

    def _lookup_title_to_pid(self, title: str) -> int | None:
        """Look up page_id for a title via SQLite (indexed, ~0.003ms)."""
        row = self.conn.execute(
            "SELECT page_id FROM canonical WHERE page_title = ?", (title,)
        ).fetchone()
        return row[0] if row else None

    def _resolve_title_to_pid(self, title: str) -> int | None:
        """Resolve a title to a canonical page_id, following redirects.

        Checks canonical first, then tries redirect resolution.
        """
        # Direct canonical lookup
        pid = self._lookup_title_to_pid(title)
        if pid is not None:
            return pid

        # Try as a redirect
        row = self.conn.execute(
            "SELECT page_id FROM page WHERE page_title = ? AND page_ns = 0 AND is_redirect = 1",
            (title,),
        ).fetchone()
        if row:
            rd_pid = row[0]
            if rd_pid < len(self._redirect_arr):
                canonical_pid = int(self._redirect_arr[rd_pid])
                if canonical_pid >= 0:
                    return canonical_pid
        return None

    def _make_observation(self, state: EpisodeState, episode_id: str) -> PageObservation:
        pid = state.current_page_id
        idx = self._lookup_pid_to_idx(pid)
        title = self._idx_to_title.get(idx, '') if idx is not None else ''

        # Get full article text or fall back to lead_text from DB
        article_text = None
        if self._article_reader is not None:
            article_text = self._article_reader.get_article_text(pid)

        if article_text is None:
            # Fallback to cached lead text
            row = self.conn.execute(
                "SELECT lead_text FROM lead_text WHERE page_id = ?", (pid,)
            ).fetchone()
            article_text = row[0] if row else ""

        # Parse into sections
        parsed = parse_article_sections(article_text)
        self._last_parsed[episode_id] = parsed

        # Get legal click set from DB (cached for get_section filtering)
        legal_links = self._get_page_links(pid)
        self._last_legal_links[episode_id] = legal_links

        # Build section summaries and links_by_section (filtered to legal links)
        # Global budget: cap total links in the observation to keep it under
        # tool output limits. Each link with context is ~180 chars; 200 links
        # ≈ 36K chars, well within any reasonable limit.
        max_links_per_section = 50
        max_links_total = 200
        section_summaries: list[SectionSummary] = []
        links_by_section: dict[str, list[dict]] = {}
        total_link_count = 0
        links_budget = max_links_total

        for sec in parsed.sections:
            # Intersect section's wikitext links with the legal click set
            legal_in_section = [
                {"title": lc.title, "context": lc.context}
                for lc in sec.links if lc.title in legal_links
            ]
            n_legal = len(legal_in_section)
            total_link_count += n_legal
            section_summaries.append(SectionSummary(
                name=sec.name, level=sec.level, link_count=n_legal))
            if legal_in_section and links_budget > 0:
                cap = min(n_legal, max_links_per_section, links_budget)
                links_by_section[sec.name] = legal_in_section[:cap]
                if cap < n_legal:
                    links_by_section[sec.name].append({
                        "note": f"...and {n_legal - cap} more links (use get_section to see all)"
                    })
                links_budget -= cap

        # Lead paragraph: first ~5000 chars of the Lead section
        # (captures the full lead for ~95% of articles)
        lead_paragraph = ''
        lead_truncated = False
        for sec in parsed.sections:
            if sec.name == 'Lead':
                if len(sec.text) <= 5000:
                    lead_paragraph = sec.text
                else:
                    lead_truncated = True
                    lead_paragraph = sec.text[:5000]
                    # Trim to last sentence boundary
                    cut = lead_paragraph.rfind('. ')
                    if cut > 2000:
                        lead_paragraph = lead_paragraph[:cut + 1]
                break

        path_titles = [
            self._idx_to_title.get(i, f"?{p}") if (i := self._lookup_pid_to_idx(p)) is not None else f"?{p}"
            for p in state.path
        ]

        return PageObservation(
            episode_id=episode_id,
            snapshot=self.snapshot,
            target_title=state.episode.target_title,
            current_title=title,
            infobox=parsed.infobox,
            lead_paragraph=lead_paragraph,
            lead_truncated=lead_truncated,
            sections=section_summaries,
            links_by_section=links_by_section,
            total_link_count=total_link_count,
            clicks_so_far=state.clicks,
            step_limit=state.episode.step_limit,
            path_so_far=path_titles,
        )

    def _get_page_links(self, page_id: int) -> set[str]:
        """Get the legal click set for a page.

        Uses pre-extracted visible links from page_links table,
        filtered to titles that resolve to canonical articles
        (either directly or through redirects).
        """
        links: set[str] = set()
        for row in self.conn.execute(
            "SELECT link_title FROM page_links WHERE page_id = ?", (page_id,)
        ):
            title = row[0]
            pid = self._resolve_title_to_pid(title)
            if pid is not None:
                links.add(title)
        return links

    def _expire_idle(self) -> None:
        now = time.time()
        expired = [
            eid for eid, state in self._episodes.items()
            if now - state.last_active_at > EPISODE_EXPIRY
        ]
        for eid in expired:
            state = self._episodes[eid]
            if not state.terminated:
                state.terminated = True
                state.terminated_reason = 'expired'
            del self._episodes[eid]


def create_mcp_server(
    db_path: Path,
    graph_dir: Path,
    episodes_path: Path,
    dump_path: Path | None = None,
    index_path: Path | None = None,
    snapshot: str = "enwiki-latest",
):
    """Create and return an MCP server with wiki benchmark tools.

    Requires the 'mcp' package to be installed.
    """
    from mcp.server.fastmcp import FastMCP

    engine = WikiBenchEngine(
        db_path, graph_dir, episodes_path,
        dump_path=dump_path, index_path=index_path,
        snapshot=snapshot,
    )
    mcp = FastMCP("wikibench")

    @mcp.tool()
    def start_episode(seed: int | None = None) -> str:
        """Start a new Wikipedia racing episode.

        Args:
            seed: Index into the pre-sampled episode catalog. If None, uses episode 0.

        Returns:
            JSON with episode_id, start/target titles, step_limit, and initial
            observation. The observation contains: infobox, lead_paragraph,
            sections (table of contents with link counts), and links_by_section
            (clickable links grouped by article section).
        """
        result = engine.start_episode(seed)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def get_page(episode_id: str) -> str:
        """Get the current page observation.

        Args:
            episode_id: The episode identifier.

        Returns:
            JSON with current_title, infobox, lead_paragraph, sections (table of
            contents), links_by_section (grouped clickable links), clicks_so_far,
            step_limit, path_so_far.
        """
        result = engine.get_page(episode_id)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def get_section(episode_id: str, section: str) -> str:
        """Read the full text of a specific article section.

        Use this to read a section in detail when the table of contents shows
        it has relevant links. Like scrolling to a section on the page.

        Args:
            episode_id: The episode identifier.
            section: Section name (from the sections list in the observation).

        Returns:
            JSON with section name, full text, and links found in that section.
        """
        result = engine.get_section(episode_id, section)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def click_link(episode_id: str, title: str) -> str:
        """Click a link on the current page to navigate to it.

        Args:
            episode_id: The episode identifier.
            title: The title of the article to navigate to. Must appear in
                   links_by_section.

        Returns:
            JSON with ok (bool), done (bool), optional error, and new observation.
        """
        result = engine.click_link(episode_id, title)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def search_page(episode_id: str, query: str, max_results: int = 10) -> str:
        """Search the current article text for a string (like ctrl+F).

        Returns matching lines with surrounding context. Useful for finding
        relevant links without reading the entire article.

        Args:
            episode_id: The episode identifier.
            query: Text to search for (case-insensitive).
            max_results: Maximum number of matches to return (default 10).

        Returns:
            JSON with query, total_matches count, and list of matches with
            line_number and surrounding context.
        """
        result = engine.search_page(episode_id, query, max_results)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def score_episode(episode_id: str) -> str:
        """Score a completed episode.

        Args:
            episode_id: The episode identifier.

        Returns:
            JSON with success, clicks, path, shortest_path_len, optimality_gap, etc.
        """
        result = engine.score_ep(episode_id)
        return json.dumps(result, indent=2)

    return mcp
