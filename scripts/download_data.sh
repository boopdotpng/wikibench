#!/usr/bin/env bash
set -euo pipefail

# Download all data required to build the wikibench benchmark.
#
# Requirements:
#   - aria2c (for the multistream XML dump via BitTorrent)
#   - curl or wget (for the smaller SQL dumps via HTTPS)
#
# Usage:
#   ./scripts/download_data.sh              # download everything
#   ./scripts/download_data.sh --skip-xml   # skip the 25 GB XML dump (if you already have it)

OUT_DIR="${WIKIBENCH_DATA_DIR:-data/raw}"
SKIP_XML=false
DUMP_DATE="20260101"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-xml)  SKIP_XML=true ;;
        --out-dir)   OUT_DIR="$2"; shift ;;
        --out-dir=*) OUT_DIR="${1#*=}" ;;
        *)           echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUT_DIR"

# --- Helper ---
download_https() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  Already exists: $(basename "$dest"), skipping"
        return
    fi
    echo "  Downloading $(basename "$dest")..."
    curl -L --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
}

# --- 1. SQL dumps via HTTPS (fast, ~10 GB total) ---
BASE="https://dumps.wikimedia.org/enwiki/latest"

echo "=== Downloading SQL dumps ==="
download_https "$BASE/enwiki-latest-page.sql.gz"        "$OUT_DIR/enwiki-${DUMP_DATE}-page.sql.gz"
download_https "$BASE/enwiki-latest-pagelinks.sql.gz"    "$OUT_DIR/enwiki-${DUMP_DATE}-pagelinks.sql.gz"
download_https "$BASE/enwiki-latest-linktarget.sql.gz"   "$OUT_DIR/enwiki-${DUMP_DATE}-linktarget.sql.gz"
download_https "$BASE/enwiki-latest-redirect.sql.gz"     "$OUT_DIR/enwiki-${DUMP_DATE}-redirect.sql.gz"
download_https "$BASE/enwiki-latest-page_props.sql.gz"   "$OUT_DIR/enwiki-${DUMP_DATE}-page_props.sql.gz"
echo ""

# --- 2. Multistream index via HTTPS (small, ~265 MB) ---
echo "=== Downloading multistream index ==="
download_https "$BASE/enwiki-latest-pages-articles-multistream-index.txt.bz2" \
    "$OUT_DIR/enwiki-${DUMP_DATE}-pages-articles-multistream-index.txt.bz2"
echo ""

# --- 3. Multistream XML dump via BitTorrent (~25 GB) ---
if [ "$SKIP_XML" = true ]; then
    echo "=== Skipping XML dump (--skip-xml) ==="
else
    XML_FILE="$OUT_DIR/enwiki-${DUMP_DATE}-pages-articles-multistream.xml.bz2"
    if [ -f "$XML_FILE" ]; then
        echo "=== XML dump already exists, skipping ==="
    else
        echo "=== Downloading XML dump via BitTorrent (aria2c) ==="
        if ! command -v aria2c &>/dev/null; then
            echo "ERROR: aria2c is required for the BitTorrent download." >&2
            echo "Install it with: sudo apt install aria2  (or brew install aria2)" >&2
            echo "Or download the XML dump manually and place it at:" >&2
            echo "  $XML_FILE" >&2
            exit 1
        fi

        MAGNET="magnet:?xt=urn:btih:e7d78d128db80266830e64c0142a67d0c5413ced&dn=enwiki-20260101-pages-articles-multistream.xml.bz2&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337%2fannounce&tr=https%3a%2f%2fipv6.academictorrents.com%2fannounce.php&tr=https%3a%2f%2facademictorrents.com%2fannounce.php"

        aria2c \
            --seed-time=0 \
            --max-upload-limit=1M \
            --dir="$OUT_DIR" \
            --out="enwiki-${DUMP_DATE}-pages-articles-multistream.xml.bz2" \
            "$MAGNET"
    fi
fi
echo ""

# --- 4. Create enwiki-latest-* symlinks ---
echo "=== Creating enwiki-latest symlinks ==="
cd "$OUT_DIR"
for f in enwiki-${DUMP_DATE}-*; do
    link="enwiki-latest-${f#enwiki-${DUMP_DATE}-}"
    if [ ! -L "$link" ]; then
        ln -sf "$f" "$link"
        echo "  $link -> $f"
    fi
done
cd - >/dev/null
echo ""

echo "=== Download complete ==="
echo "Files are in: $OUT_DIR"
echo ""
echo "Next steps:"
echo "  ./scripts/run_pipeline.sh --skip-download"
