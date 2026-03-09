"""Data schemas for the Wikipedia racing benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


@dataclass
class Episode:
    """A pre-sampled benchmark episode."""
    episode_id: str
    split: str  # 'dev' | 'test'
    seed: int
    start_page_id: int
    target_page_id: int
    start_title: str
    target_title: str
    shortest_path_len: int
    difficulty: str  # 'easy' | 'medium' | 'hard' | 'nightmare'
    step_limit: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> Episode:
        return cls(**json.loads(s))


@dataclass
class SectionSummary:
    """Compact section info for the table of contents."""
    name: str
    level: int
    link_count: int


@dataclass
class PageObservation:
    """What the model sees on each turn."""
    episode_id: str
    snapshot: str
    target_title: str
    current_title: str
    infobox: str
    lead_paragraph: str
    lead_truncated: bool
    sections: list[SectionSummary]
    links_by_section: dict[str, list[dict]]
    total_link_count: int
    clicks_so_far: int
    step_limit: int
    path_so_far: list[str]


@dataclass
class EpisodeState:
    """Server-side mutable state for a running episode."""
    episode: Episode
    current_page_id: int
    path: list[int] = field(default_factory=list)
    clicks: int = 0
    invalid_actions: int = 0
    terminated: bool = False
    terminated_reason: (
        Literal['success', 'step_limit', 'invalid_action_limit', 'aborted', 'expired']
        | None
    ) = None
    created_at: float = 0.0
    last_active_at: float = 0.0


@dataclass
class ScoreResult:
    """Final scoring output for a completed episode."""
    success: bool
    clicks: int
    path: list[str]
    shortest_path_len: int | None
    optimality_gap: int | None
    invalid_actions: int
    terminated_reason: str


def write_episodes_jsonl(episodes: list[Episode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for ep in episodes:
            f.write(ep.to_json() + '\n')


def load_episodes_jsonl(path: Path) -> list[Episode]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(Episode.from_json(line))
    return episodes
