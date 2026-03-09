"""Shared wikitext cleanup helpers."""

from __future__ import annotations

import re

_RE_REF = re.compile(r'<ref[^>]*/>', re.DOTALL)
_RE_REF_BLOCK = re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL)
_RE_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)
_RE_GALLERY = re.compile(r'<gallery[^>]*>.*?</gallery>', re.DOTALL)
_RE_MATH = re.compile(r'<math[^>]*>.*?</math>', re.DOTALL)
_RE_NOWIKI = re.compile(r'<nowiki[^>]*>.*?</nowiki>', re.DOTALL)
_RE_TEMPLATE = re.compile(r'\{\{[^{}]*\}\}')

_STRIP_BLOCK_PATTERNS = (
    _RE_REF,
    _RE_REF_BLOCK,
    _RE_COMMENT,
    _RE_GALLERY,
    _RE_MATH,
    _RE_NOWIKI,
)


def normalize_title(title: str) -> str:
    """Normalize a wiki title for lookups and comparisons."""
    title = title.strip().replace(' ', '_')
    return title[:1].upper() + title[1:] if title else title


def strip_tag_blocks(text: str) -> str:
    """Remove common non-visible tag regions from wikitext."""
    for pattern in _STRIP_BLOCK_PATTERNS:
        text = pattern.sub('', text)
    return text


def strip_templates(text: str, *, max_passes: int = 10) -> str:
    """Strip simple/nested templates with bounded repeated passes."""
    for _ in range(max_passes):
        new = _RE_TEMPLATE.sub('', text)
        if new == text:
            return text
        text = new
    return text


def strip_navbox_region(wikitext: str, prefixes: tuple[str, ...]) -> str:
    """Remove navbox/footer template regions from the end of wikitext."""
    lines = wikitext.split('\n')
    cutoff = len(lines)

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip().lower()
        if stripped.startswith('{{'):
            template_name = stripped[2:].split('|')[0].split('}')[0].strip()
            if any(template_name.startswith(prefix) for prefix in prefixes):
                cutoff = i
                continue
        if stripped == '' or stripped.startswith('[[category:'):
            cutoff = i
            continue
        break

    return '\n'.join(lines[:cutoff])
