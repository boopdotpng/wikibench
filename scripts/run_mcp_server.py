"""Run the Wikipedia benchmark MCP server."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option('--db-path', default='data/processed/wiki.db',
              type=click.Path(exists=True))
@click.option('--graph-dir', default='data/processed',
              type=click.Path(exists=True))
@click.option('--episodes', default='episodes/dev.jsonl',
              type=click.Path(exists=True))
@click.option('--dump-path', default='data/raw/enwiki-latest-pages-articles-multistream.xml.bz2',
              type=click.Path(exists=True))
@click.option('--index-path', default='data/raw/enwiki-latest-pages-articles-multistream-index.txt.bz2',
              type=click.Path(exists=True))
@click.option('--snapshot', default='enwiki-latest')
@click.option('--transport', type=click.Choice(['stdio', 'sse']), default='stdio')
def main(db_path: str, graph_dir: str, episodes: str, dump_path: str,
         index_path: str, snapshot: str, transport: str) -> None:
    """Start the Wikipedia benchmark MCP server."""
    from wikipedia_bench.mcp_server import create_mcp_server

    mcp = create_mcp_server(
        db_path=Path(db_path),
        graph_dir=Path(graph_dir),
        episodes_path=Path(episodes),
        dump_path=Path(dump_path),
        index_path=Path(index_path),
        snapshot=snapshot,
    )

    mcp.run(transport=transport)


if __name__ == '__main__':
    main()
