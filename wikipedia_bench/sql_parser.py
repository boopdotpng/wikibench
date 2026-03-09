"""Streaming parser for MediaWiki SQL dump files.

Handles the massive INSERT INTO statements found in Wikimedia database dumps.
Reads compressed (.gz) files in chunks and parses VALUES tuples via a byte-level
state machine with regex-accelerated scanning for performance.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB read chunks

# Regex patterns for fast scanning within buffers.
# When NOT in a string, we care about: single-quote (string start), comma
# (field separator), close-paren (tuple end).
_RE_OUTSIDE_STR = re.compile(rb"[',)]")
# When IN a string, we care about: single-quote (string end or '' escape),
# backslash (escape sequence start).
_RE_INSIDE_STR = re.compile(rb"['\\\\]")

_ESCAPE_MAP = {
    ord('\\'): ord('\\'),
    ord("'"): ord("'"),
    ord('n'): ord('\n'),
    ord('r'): ord('\r'),
    ord('0'): 0,
    ord('Z'): 0x1a,
    ord('t'): ord('\t'),
}


class _Buf:
    """Buffer manager for streaming through large binary files.

    Maintains a bytearray that is periodically compacted. Provides methods
    for pattern scanning and refilling from the underlying stream.
    """

    __slots__ = ('_f', '_data', '_pos', '_eof')

    def __init__(self, f: BinaryIO):
        self._f = f
        self._data = bytearray()
        self._pos = 0
        self._eof = False

    def _ensure(self, need: int = 1) -> bool:
        """Ensure at least `need` bytes available from _pos. Returns False at EOF."""
        while (len(self._data) - self._pos) < need:
            if self._eof:
                return (len(self._data) - self._pos) >= need
            if self._pos > CHUNK_SIZE:
                del self._data[:self._pos]
                self._pos = 0
            chunk = self._f.read(CHUNK_SIZE)
            if not chunk:
                self._eof = True
            else:
                self._data.extend(chunk)
        return True

    def scan_to(self, pattern: bytes) -> bool:
        """Advance past the next occurrence of `pattern`. Returns False at EOF."""
        plen = len(pattern)
        while True:
            self._ensure(plen)
            idx = self._data.find(pattern, self._pos)
            if idx != -1:
                self._pos = idx + plen
                return True
            if self._eof:
                return False
            self._pos = max(self._pos, len(self._data) - plen + 1)

    def read_tuple(self, keep_columns: tuple[int, ...] | None = None) -> list | None:
        """Read next VALUES tuple. Returns None at end of INSERT (;) or EOF.

        When keep_columns is provided, only those 0-based field indices are
        materialized, in the same order as keep_columns. Unselected fields are
        still parsed correctly but skipped without buffer allocation.
        """
        # Skip to opening '(' or terminating ';'
        while True:
            if not self._ensure(1):
                return None
            b = self._data[self._pos]
            if b == ord(';'):
                self._pos += 1
                return None
            if b == ord('('):
                self._pos += 1
                break
            self._pos += 1

        # Parse fields until closing ')'
        if keep_columns is None:
            fields: list = []
            keep_map: dict[int, int] | None = None
        else:
            fields = [None] * len(keep_columns)
            keep_map = {col: idx for idx, col in enumerate(keep_columns)}

        field_idx = 0
        capture_idx = None if keep_map is None else keep_map.get(field_idx)
        field_buf = bytearray() if capture_idx is not None or keep_map is None else None
        in_string = False
        is_str = False

        while True:
            if not self._ensure(1):
                return None

            if in_string:
                match = _RE_INSIDE_STR.search(self._data, self._pos)
                if match is None:
                    # All remaining buffer is string content; refill
                    if field_buf is not None:
                        field_buf.extend(self._data[self._pos:])
                    self._pos = len(self._data)
                    if not self._ensure(1):
                        return None
                    continue

                if field_buf is not None:
                    field_buf.extend(self._data[self._pos:match.start()])
                self._pos = match.start() + 1
                ch = self._data[match.start()]

                if ch == ord('\\'):
                    if not self._ensure(1):
                        return None
                    esc = self._data[self._pos]
                    self._pos += 1
                    if field_buf is not None:
                        field_buf.append(_ESCAPE_MAP.get(esc, esc))
                else:  # single quote
                    # Check for '' (SQL-standard doubled quote)
                    if (self._ensure(1)
                            and self._pos < len(self._data)
                            and self._data[self._pos] == ord("'")):
                        if field_buf is not None:
                            field_buf.append(ord("'"))
                        self._pos += 1
                    else:
                        in_string = False
            else:
                match = _RE_OUTSIDE_STR.search(self._data, self._pos)
                if match is None:
                    if field_buf is not None:
                        field_buf.extend(self._data[self._pos:])
                    self._pos = len(self._data)
                    if not self._ensure(1):
                        return None
                    continue

                if field_buf is not None and match.start() > self._pos:
                    field_buf.extend(self._data[self._pos:match.start()])
                self._pos = match.start() + 1
                ch = self._data[match.start()]

                if ch == ord("'"):
                    in_string = True
                    is_str = field_buf is not None
                elif ch == ord(','):
                    if field_buf is not None:
                        value = _finish_field(field_buf, is_str)
                        if keep_map is None:
                            fields.append(value)
                        else:
                            fields[capture_idx] = value
                    field_idx += 1
                    capture_idx = None if keep_map is None else keep_map.get(field_idx)
                    field_buf = bytearray() if capture_idx is not None or keep_map is None else None
                    is_str = False
                elif ch == ord(')'):
                    if field_buf is not None:
                        value = _finish_field(field_buf, is_str)
                        if keep_map is None:
                            fields.append(value)
                        else:
                            fields[capture_idx] = value
                    return fields


def _finish_field(buf: bytearray, is_string: bool):
    """Convert a raw field buffer to a Python value."""
    if is_string:
        return buf.decode('utf-8', errors='replace')
    if not buf:
        return None
    if buf == b'NULL':
        return None
    try:
        return int(buf)
    except ValueError:
        try:
            return float(buf)
        except ValueError:
            return buf.decode('utf-8', errors='replace')


class _ProgressWrapper:
    """Wraps a raw file object to feed byte counts to tqdm."""

    def __init__(self, f: BinaryIO, pbar):
        self._f = f
        self._pbar = pbar

    def read(self, n: int = -1) -> bytes:
        data = self._f.read(n)
        self._pbar.update(len(data))
        return data

    def readinto(self, b) -> int | None:
        n = self._f.readinto(b)
        if n:
            self._pbar.update(n)
        return n

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self):
        pass


def iter_sql_rows(
    path: Path | str,
    table_name: str,
    *,
    progress: bool = True,
    keep_columns: Iterable[int] | None = None,
) -> Iterator[list]:
    """Stream-parse a MediaWiki SQL dump, yielding parsed row lists.

    Args:
        path: Path to .sql.gz (or .sql) dump file.
        table_name: MySQL table name to extract (e.g. 'page', 'pagelinks').
        progress: Show tqdm progress bar (tracks compressed bytes read).
        keep_columns: Optional 0-based field indices to materialize. When
            provided, yielded lists contain only those columns, in the same
            order. This is useful for high-volume imports that only need a few
            fields from a wide table.

    Yields:
        list of Python values (int, float, str, None) for each row tuple.
    """
    path = Path(path)
    marker = f"INSERT INTO `{table_name}` VALUES ".encode('ascii')

    keep_columns_tuple = tuple(keep_columns) if keep_columns is not None else None

    if path.suffix == '.gz':
        raw_f = open(path, 'rb')
        if progress:
            from tqdm import tqdm
            pbar = tqdm(
                total=path.stat().st_size,
                unit='B',
                unit_scale=True,
                desc=table_name,
            )
            wrapped = _ProgressWrapper(raw_f, pbar)
            f = gzip.open(wrapped, 'rb')  # type: ignore[arg-type]
        else:
            pbar = None
            f = gzip.open(raw_f, 'rb')
    else:
        pbar = None
        raw_f = None
        f = open(path, 'rb')

    try:
        buf = _Buf(f)
        while buf.scan_to(marker):
            while True:
                row = buf.read_tuple(keep_columns_tuple)
                if row is None:
                    break
                yield row
    finally:
        f.close()
        if raw_f is not None:
            raw_f.close()
        if pbar is not None:
            pbar.close()
