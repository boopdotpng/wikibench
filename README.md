# wikibench

A Wikipedia racing benchmark for LLMs. The model starts at a random Wikipedia article and must navigate to a target article by clicking links, using as few clicks as possible.

The benchmark runs entirely offline against a frozen English Wikipedia dump. Models interact through an [MCP](https://modelcontextprotocol.io/) server that exposes six tools: `start_episode`, `click_link`, `get_section`, `get_page`, `search_page`, and `score_episode`.

**~4,100 lines of Python.** Built from the `enwiki-20260101` dump (6.74M articles, 682M links).

## What it measures

Local semantic navigation — can the model read an article, reason about which linked concept is closest to the target, and get there efficiently? The model sees the full article text with section headers, infoboxes, and inline `[[wikilinks]]`, the same information a human player would see.

## Quick start

### 1. Download data (~37 GB)

```bash
./scripts/download_data.sh
```

The XML dump (~25 GB) downloads via BitTorrent (requires `aria2c`). Everything else uses HTTPS.

### 2. Build the benchmark

```bash
./scripts/run_pipeline.sh --skip-download
```

This imports SQL dumps into SQLite, builds the link graph, extracts article text, and samples episodes. Takes 2-4 hours depending on disk speed. Requires ~150 GB free disk space.

### 3. Run the MCP server

```bash
uv run python3 scripts/run_mcp_server.py
```

Startup takes ~1s (cached) or ~50s (first run, builds binary caches). The server communicates over stdio.

### 4. Play

Connect any MCP-compatible client (Claude Code, Codex, etc.) and start an episode:

```
start_episode(seed=0)
```

The server returns the start article, target article, step limit, and the full page observation. Click links to navigate:

```
click_link(episode_id="...", title="United_States")
```

## MCP configuration

### Claude Code

The repo includes `.mcp.json`:

```json
{
  "mcpServers": {
    "wikibench": {
      "command": "uv",
      "args": ["run", "python3", "scripts/run_mcp_server.py"],
      "cwd": "/path/to/wikibench",
      "timeout": 120,
      "env": { "PYTHONPATH": "." }
    }
  }
}
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.wikibench]
command = "uv"
args = ["run", "python3", "scripts/run_mcp_server.py"]
cwd = "/path/to/wikibench"
startup_timeout_sec = 60
env = { PYTHONPATH = "." }
```

## MCP tools

| Tool | Description |
|------|-------------|
| `start_episode(seed)` | Start a new episode. Returns episode_id, start/target titles, step_limit, and the initial page observation. |
| `click_link(episode_id, title)` | Click a link on the current page. Title must exactly match one from `links_by_section`. |
| `get_section(episode_id, section)` | Read the full text of a specific section. Like scrolling to a section on the page. |
| `get_page(episode_id)` | Re-read the current page observation. |
| `search_page(episode_id, query)` | Search the current article text for a string (case-insensitive). Like ctrl+F. |
| `score_episode(episode_id)` | Get the final score after the episode ends. |

Each page observation includes: an infobox with structured metadata, a lead paragraph, a table of contents with link counts per section, all clickable links grouped by section (`links_by_section`), your click count and step limit, and your path so far. The model can use `get_section` to read full section text on demand.

## Scoring

| Metric | Description |
|--------|-------------|
| `success` | Reached the target within the step limit |
| `clicks` | Number of link clicks taken |
| `shortest_path_len` | Optimal path length in the frozen graph |
| `optimality_gap` | `clicks - shortest_path_len` (lower is better) |
| `invalid_actions` | Links clicked that weren't on the page |

Step limit: `max(2 * shortest_path_len + 4, 12)`

## Episode difficulty

| Difficulty | Shortest path |
|------------|---------------|
| Easy | 2-3 clicks |
| Medium | 4-5 clicks |
| Hard | 6+ clicks |

## Project structure

```
wikibench/
  spec.md                    # design spec
  wikibench-start-prompt.md  # system prompt for the model
  .mcp.json                  # MCP server config
  pyproject.toml
  wikipedia_bench/           # core library
    schemas.py               # dataclasses (Episode, PageObservation, etc.)
    db.py                    # SQLite schema and import
    redirects.py             # redirect resolution
    graph.py                 # CSR graph with mmap, BFS
    text_extract.py          # multistream article extraction
    article_reader.py        # on-the-fly article reader with infobox parsing
    sampler.py               # episode sampling
    scorer.py                # episode scoring
    mcp_server.py            # MCP server (WikiBenchEngine + FastMCP)
    sql_parser.py            # streaming SQL dump parser
  scripts/
    download_data.sh         # download all dump files
    run_pipeline.sh          # full build pipeline
    download_dumps.py        # Python downloader (alternative)
    build_db.py              # import SQL dumps into SQLite
    build_graph.py           # build CSR graph
    build_text_cache.py      # extract lead text + links
    sample_episodes.py       # sample episodes
    run_mcp_server.py        # launch MCP server
    evaluate.py              # batch evaluation harness
  episodes/                  # pre-sampled episode splits (.jsonl)
  tests/
    test_integration.py      # full pipeline integration test
    test_sql_parser.py       # SQL parser unit tests
  data/
    raw/                     # downloaded dump files (~37 GB)
    processed/               # wiki.db + graph arrays (~95 GB)
```

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- ~150 GB free disk space
- `aria2c` for the BitTorrent download (or download the XML dump manually)

## Dependencies

- `numpy` - graph arrays and mmap
- `click` - CLI scripts
- `tqdm` - progress bars
- `mwparserfromhell` - wikitext parsing
- `mcp` - MCP server framework

## Tests

```bash
uv run python3 -m pytest tests/
```

## License

The Wikipedia dump data is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
