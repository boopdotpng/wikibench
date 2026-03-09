# wikibench spec

status: v1.0
last updated: 2026-03-09

## 1. goal

A benchmark where an LLM plays Wikipedia racing the way a human would:

- start at a random Wikipedia article
- reach a target article by clicking links
- minimize clicks
- observe only the current page (not the whole graph)
- use visible page information (infobox, section outline, links grouped by section)

Deterministic, offline, replayable. The model interacts through an MCP server.

## 2. non-goals

- Not a browser automation benchmark. No live browser, DOM, or public website UI.
- Not a blind graph-search benchmark where the model sees only link titles with no article text.

## 3. what the model sees

On each turn, the model receives a compact page observation:

```json
{
  "episode_id": "uuid",
  "snapshot": "enwiki-latest",
  "target_title": "Erika_Heynatz",
  "current_title": "Lori_Daniels",
  "infobox": "[Infobox: person]\n  name: Lori Daniels\n  ...",
  "lead_paragraph": "Lori Daniels (born November 5, 1955) is a former member of the...",
  "lead_truncated": false,
  "sections": [
    {"name": "Infobox", "level": 0, "link_count": 3},
    {"name": "Lead", "level": 0, "link_count": 1},
    {"name": "Career", "level": 2, "link_count": 12},
    {"name": "Personal life", "level": 2, "link_count": 5}
  ],
  "links_by_section": {
    "Infobox": [
      {"title": "United_States", "context": "...birth_place: Los Angeles, California\n  nationality: United States..."},
      {"title": "Los_Angeles", "context": "...birth_place: Los Angeles, California\n  nationality:..."}
    ],
    "Lead": [
      {"title": "Actress", "context": "...is an American actress and singer who appeared in..."}
    ]
  },
  "total_link_count": 21,
  "clicks_so_far": 0,
  "step_limit": 12,
  "path_so_far": ["Lori_Daniels"]
}
```

This mirrors how a human plays: glance at the page, see which sections have relevant links, drill into promising sections. The model can:

- See the infobox sidebar (often the most link-dense part)
- See a lead paragraph (~5000 chars, covers the full lead for most articles; `lead_truncated` flag if cut)
- See a table of contents with link density per section
- See clickable links grouped by section, each with a context snippet (~10 words before and after the link, showing why it appears)
- See `total_link_count` to gauge article size at a glance
- Use `get_section` to read the full text of a specific section
- Use `search_page` to ctrl+F for keywords

Sections with more than 50 links are capped, with a note to use `get_section` for the full list.

The model does not receive: the full article text by default, hidden summaries of linked pages, the full text of prior pages, or any graph metadata.

## 4. MCP tools

```
start_episode(seed?: int)
  Start a new episode. Returns episode_id, start/target titles,
  step_limit, and the initial page observation.

click_link(episode_id, title)
  Click a link on the current page. Title must exactly match
  one from links_by_section. Returns ok, done, and new observation.

get_section(episode_id, section)
  Read the full text of a specific section. Like scrolling to
  a section on the page.

get_page(episode_id)
  Re-read the current page observation.

search_page(episode_id, query, max_results?)
  Search the current article text for a string (case-insensitive).
  Returns matching lines with surrounding context. Like ctrl+F.

score_episode(episode_id)
  Return final score after the episode ends.
```

## 5. offline data source

English Wikipedia only. Built from official Wikimedia dump files (enwiki-20260101):

| File | Size | Purpose |
|------|------|---------|
| page.sql.gz | 2.4 GB | Page IDs, titles, namespaces |
| pagelinks.sql.gz | 6.9 GB | Internal links |
| linktarget.sql.gz | 1.4 GB | Normalized link targets |
| redirect.sql.gz | 0.2 GB | Redirect mappings |
| page_props.sql.gz | 0.4 GB | Disambiguation detection |
| multistream.xml.bz2 | 25 GB | Full article content |
| multistream-index.txt.bz2 | 0.3 GB | Byte-offset index for article lookup |

## 6. world construction

### page universe

6.74M canonical articles. Namespace 0 only. Excludes redirects and disambiguation pages from playable states.

### redirect resolution

All links are resolved through redirect chains to canonical destinations. Broken redirects and cycles are handled safely.

### link graph

Directed CSR graph: 6.74M nodes, 682.7M edges. Stored as mmap'd int32 arrays (~5.25 GB). Edges come from pagelinks joined through linktarget, resolved through redirects, filtered to canonical namespace-0 targets, self-loops removed.

### article text

Full article text is read on-the-fly from the multistream XML dump using a byte-offset index. Wikitext is cleaned to preserve section headers and inline `[[wikilinks]]` while stripping templates, refs, galleries, HTML tags, and navboxes. Infoboxes are extracted and rendered as structured key-value pairs. Articles are parsed into sections with per-section link extraction.

## 7. episode sampling

The benchmark uses a single frozen `benchmark.jsonl` dataset rather than separate dev/test splits.
Pairs are sampled from the frozen graph with a walk-first strategy:

- easy targets shortest paths of 3-4
- medium targets shortest path 5
- a tiny hard bonus set targets shortest paths of 6-7
- candidate targets come from short random walks
- each candidate pair is then verified with bidirectional BFS
- shortest-path verification is capped at depth 7
- hard examples bias toward low out-degree source pages and low out-degree walk neighbors to stay out of the dense graph core
- generation writes both JSONL and a review CSV with blank review fields for keep/drop/regenerate decisions

This is intentionally optimized for generation speed. Random-pair rejection sampling is cheap for 3-5 hop pairs but wastes a lot of time chasing scarce 6-7 hop examples, so the main benchmark stops at medium and the rare tail is reported separately.

| Difficulty | Shortest path | Step limit formula |
|------------|---------------|--------------------|
| Easy | 3-4 | max(2*sp + 4, 12) |
| Medium | 5 | max(2*sp + 4, 12) |

Bonus set:

| Set | Shortest path | Step limit formula |
|-----|---------------|--------------------|
| Hard bonus | 6-7 | max(3*sp + 4, 20) |

Fixed RNG seeds are used for reproducibility. The intended output is a single reviewed benchmark set plus a tiny hard bonus set; rejected episodes can be regenerated and swapped in after manual review.

## 8. scoring

| Metric | Description |
|--------|-------------|
| success | Reached target within step limit |
| clicks | Total link clicks |
| shortest_path_len | Optimal path in frozen graph (bidirectional BFS) |
| optimality_gap | clicks - shortest_path_len |
| invalid_actions | Illegal link clicks |
| terminated_reason | success / step_limit / invalid_action_limit / expired |

### termination

- target reached
- step limit exceeded
- 3 invalid actions
- idle timeout (30 minutes)

## 9. legality

A `click_link(title)` is legal only if the title appears in `links_by_section` and resolves to a canonical article. Illegal clicks increment the invalid action counter without moving the agent.

## 10. performance

Startup time: ~1s (cached), ~10s (first run). Binary caches for graph arrays (mmap), multistream index (mmap), and lookup tables (mmap + pickle). Per-request latency: <100ms.
