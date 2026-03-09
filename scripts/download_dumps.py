"""Download required Wikipedia dump files from Wikimedia."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import click
from tqdm import tqdm

BASE_URL = "https://dumps.wikimedia.org/enwiki/latest"

REQUIRED_FILES = [
    "enwiki-latest-page.sql.gz",
    "enwiki-latest-pagelinks.sql.gz",
    "enwiki-latest-linktarget.sql.gz",
    "enwiki-latest-redirect.sql.gz",
    "enwiki-latest-pages-articles-multistream.xml.bz2",
    "enwiki-latest-pages-articles-multistream-index.txt.bz2",
]

OPTIONAL_FILES = [
    "enwiki-latest-page_props.sql.gz",
]


def download_file(url: str, dest: Path, *, resume: bool = True) -> None:
    """Download a file with progress bar and optional resume support."""
    headers = {}
    mode = 'wb'
    initial = 0

    if resume and dest.exists():
        initial = dest.stat().st_size
        headers['Range'] = f'bytes={initial}-'
        mode = 'ab'

    req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range not satisfiable = already complete
            print(f"  Already complete: {dest.name}")
            return
        raise

    # Get total size
    content_length = resp.headers.get('Content-Length')
    if content_length:
        total = int(content_length) + initial
    else:
        total = None

    with tqdm(
        total=total,
        initial=initial,
        unit='B',
        unit_scale=True,
        desc=dest.name,
    ) as pbar:
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


@click.command()
@click.option('--out-dir', default='data/raw', type=click.Path(),
              help='Directory to save dump files')
@click.option('--resume/--no-resume', default=True,
              help='Resume partial downloads')
@click.option('--include-text/--no-text', default=True,
              help='Include the large article text dump (~26 GB)')
@click.option('--include-optional/--no-optional', default=True,
              help='Include optional files (page_props)')
def main(out_dir: str, resume: bool, include_text: bool, include_optional: bool) -> None:
    """Download required Wikipedia dump files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = list(REQUIRED_FILES)
    if not include_text:
        files = [f for f in files if 'multistream' not in f]
    if include_optional:
        files.extend(OPTIONAL_FILES)

    print(f"Downloading {len(files)} files to {out_path}/")
    print()

    manifest = {}
    for filename in files:
        url = f"{BASE_URL}/{filename}"
        dest = out_path / filename
        print(f"Downloading {filename}...")
        download_file(url, dest, resume=resume)
        manifest[filename] = {
            'size': dest.stat().st_size,
            'downloaded_at': datetime.now(timezone.utc).isoformat(),
            'url': url,
        }
        print()

    # Write manifest
    manifest_path = out_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {manifest_path}")


if __name__ == '__main__':
    main()
