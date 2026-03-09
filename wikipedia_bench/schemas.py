"""Data schemas for the Wikipedia racing benchmark."""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


@dataclass
class Episode:
    """A pre-sampled benchmark episode."""
    episode_id: str
    split: str  # dataset tag, e.g. 'benchmark'
    seed: int
    start_page_id: int
    target_page_id: int
    start_title: str
    target_title: str
    shortest_path_len: int
    difficulty: str  # 'easy' | 'medium' | 'hard'
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
        f.writelines(f'{ep.to_json()}\n' for ep in episodes)


def write_episodes_csv(episodes: list[Episode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(ep) for ep in episodes]
    fieldnames = [
        'episode_id',
        'split',
        'seed',
        'start_page_id',
        'target_page_id',
        'start_title',
        'target_title',
        'shortest_path_len',
        'difficulty',
        'step_limit',
        'review_status',
        'review_notes',
        'replacement_for_episode_id',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row['review_status'] = ''
            row['review_notes'] = ''
            row['replacement_for_episode_id'] = ''
            writer.writerow(row)


def load_episodes_jsonl(path: Path) -> list[Episode]:
    with open(path) as f:
        return [
            Episode.from_json(line)
            for raw_line in f
            if (line := raw_line.strip())
        ]
