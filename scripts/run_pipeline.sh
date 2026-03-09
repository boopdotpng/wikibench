#!/usr/bin/env bash
set -euo pipefail

# Full build pipeline for wikibench.
# Run from the wikibench/ directory.
#
# Usage:
#   ./scripts/run_pipeline.sh              # run everything
#   ./scripts/run_pipeline.sh --skip-download  # skip download step
#   ./scripts/run_pipeline.sh --from graph  # start from graph build step

cd "$(dirname "$0")/.."
export PYTHONPATH=".:$PYTHONPATH"

SKIP_DOWNLOAD=false
START_FROM=""
PARALLEL_POST_DB=true
TEXT_WORKERS="${WIKIBENCH_TEXT_WORKERS:-0}"
TEXT_BATCH_SIZE="${WIKIBENCH_TEXT_BATCH_SIZE:-10000}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-download)
            SKIP_DOWNLOAD=true
            ;;
        --from)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --from" >&2
                exit 1
            fi
            START_FROM="$2"
            shift
            ;;
        --from=*)
            START_FROM="${1#*=}"
            ;;
        --serial-post-db)
            PARALLEL_POST_DB=false
            ;;
        --parallel-post-db)
            PARALLEL_POST_DB=true
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

should_run() {
    local step="$1"
    if [ -z "$START_FROM" ]; then
        return 0
    fi
    case "$START_FROM" in
        download) [ "$step" != "" ] ;;
        db)       [ "$step" != "download" ] ;;
        graph)    [ "$step" != "download" ] && [ "$step" != "db" ] ;;
        text)     [ "$step" = "text" ] || [ "$step" = "sample" ] || [ "$step" = "done" ] ;;
        sample)   [ "$step" = "sample" ] || [ "$step" = "done" ] ;;
        *)        return 0 ;;
    esac
}

echo "=== wikibench build pipeline ==="
echo ""

wait_for_background_jobs() {
    while [ -n "$(jobs -pr)" ]; do
        if ! wait -n; then
            local rc=$?
            jobs -pr | xargs -r kill 2>/dev/null || true
            wait || true
            return "$rc"
        fi
    done
}

# Step 1: Download
if [ "$SKIP_DOWNLOAD" = false ] && should_run "download"; then
    echo ">>> Step 1: Downloading dump files..."
    uv run python3 scripts/download_dumps.py --out-dir data/raw --include-optional
    echo ""
fi

# Step 2: Import into SQLite
if should_run "db"; then
    echo ">>> Step 2: Importing SQL dumps into SQLite..."
    uv run python3 scripts/build_db.py --raw-dir data/raw --db-path data/processed/wiki.db
    echo ""
fi

RUN_GRAPH=false
RUN_TEXT=false
if should_run "graph"; then
    RUN_GRAPH=true
fi
if should_run "text"; then
    RUN_TEXT=true
fi

if [ "$RUN_GRAPH" = true ] && [ "$RUN_TEXT" = true ] && [ "$PARALLEL_POST_DB" = true ]; then
    echo ">>> Steps 3-4: Building graph and text cache in parallel..."
    uv run python3 scripts/build_graph.py --db-path data/processed/wiki.db --out-dir data/processed &
    uv run python3 scripts/build_text_cache.py \
        --db-path data/processed/wiki.db \
        --dump-path data/raw/enwiki-latest-pages-articles-multistream.xml.bz2 \
        --index-path data/raw/enwiki-latest-pages-articles-multistream-index.txt.bz2 \
        --workers "$TEXT_WORKERS" \
        --batch-size "$TEXT_BATCH_SIZE" &
    wait_for_background_jobs
    echo ""
else
    if [ "$RUN_GRAPH" = true ]; then
        echo ">>> Step 3: Building CSR graph..."
        uv run python3 scripts/build_graph.py --db-path data/processed/wiki.db --out-dir data/processed
        echo ""
    fi

    if [ "$RUN_TEXT" = true ]; then
        echo ">>> Step 4: Building text cache (lead text + links)..."
        uv run python3 scripts/build_text_cache.py \
            --db-path data/processed/wiki.db \
            --dump-path data/raw/enwiki-latest-pages-articles-multistream.xml.bz2 \
            --index-path data/raw/enwiki-latest-pages-articles-multistream-index.txt.bz2 \
            --workers "$TEXT_WORKERS" \
            --batch-size "$TEXT_BATCH_SIZE"
        echo ""
    fi
fi

# Step 5: Sample benchmark episodes
if should_run "sample"; then
    echo ">>> Step 5: Sampling benchmark and hard bonus episodes..."
    uv run python3 scripts/sample_episodes.py \
        --db-path data/processed/wiki.db \
        --graph-dir data/processed \
        --episodes-path episodes/benchmark.jsonl \
        --hard-bonus-path episodes/hard_bonus.jsonl
    echo ""
fi

echo "=== Pipeline complete ==="
echo "To run the MCP server:"
echo "  uv run python3 scripts/run_mcp_server.py"
echo "To run the evaluation harness:"
echo "  uv run python3 scripts/evaluate.py"
