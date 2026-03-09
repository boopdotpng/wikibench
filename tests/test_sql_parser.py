"""Tests for the streaming SQL dump parser."""

import gzip
import tempfile
from pathlib import Path

from wikipedia_bench.sql_parser import iter_sql_rows


def _write_dump(content: str, *, gz: bool = True) -> Path:
    """Write a synthetic SQL dump to a temp file."""
    suffix = '.sql.gz' if gz else '.sql'
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    if gz:
        with gzip.open(f.name, 'wt', encoding='utf-8') as gf:
            gf.write(content)
    else:
        with open(f.name, 'w') as sf:
            sf.write(content)
    return Path(f.name)


def test_basic_insert():
    sql = (
        "-- comment\n"
        "DROP TABLE IF EXISTS `test`;\n"
        "CREATE TABLE `test` (id int, name varchar(255));\n"
        "INSERT INTO `test` VALUES (1,'hello'),(2,'world');\n"
    )
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 'test', progress=False))
    assert len(rows) == 2
    assert rows[0] == [1, 'hello']
    assert rows[1] == [2, 'world']


def test_null_values():
    sql = "INSERT INTO `t` VALUES (1,NULL,'foo',NULL);\n"
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert rows == [[1, None, 'foo', None]]


def test_escaped_strings():
    sql = r"INSERT INTO `t` VALUES (1,'it\'s a test'),(2,'back\\slash');" + "\n"
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert rows[0] == [1, "it's a test"]
    assert rows[1] == [2, "back\\slash"]


def test_multiple_inserts():
    sql = (
        "INSERT INTO `t` VALUES (1,'a'),(2,'b');\n"
        "INSERT INTO `t` VALUES (3,'c');\n"
    )
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert len(rows) == 3
    assert rows[2] == [3, 'c']


def test_wrong_table_skipped():
    sql = (
        "INSERT INTO `other` VALUES (1,'skip');\n"
        "INSERT INTO `target` VALUES (2,'keep');\n"
    )
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 'target', progress=False))
    assert len(rows) == 1
    assert rows[0] == [2, 'keep']


def test_float_values():
    sql = "INSERT INTO `t` VALUES (1,0.778582,'title',0);\n"
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert rows[0][0] == 1
    assert abs(rows[0][1] - 0.778582) < 1e-6
    assert rows[0][2] == 'title'


def test_empty_string():
    sql = "INSERT INTO `t` VALUES (1,'');\n"
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert rows[0] == [1, '']


def test_string_with_parens_and_commas():
    sql = "INSERT INTO `t` VALUES (1,'hello (world), test');\n"
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert rows[0] == [1, 'hello (world), test']


def test_doubled_quote_escape():
    sql = "INSERT INTO `t` VALUES (1,'it''s doubled');\n"
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert rows[0] == [1, "it's doubled"]


def test_uncompressed_sql():
    sql = "INSERT INTO `t` VALUES (1,'plain');\n"
    path = _write_dump(sql, gz=False)
    rows = list(iter_sql_rows(path, 't', progress=False))
    assert rows[0] == [1, 'plain']


def test_page_like_row():
    """Test with a row structure resembling the actual page table."""
    sql = (
        "INSERT INTO `page` VALUES "
        "(1,0,'Main_Page',0,0,0.778582,'20231201120000',NULL,12345,6789,'wikitext',NULL),"
        "(2,0,'Albert_Einstein',0,0,0.123,'20231201120000','20231201120000',67890,45678,'wikitext',NULL);\n"
    )
    path = _write_dump(sql)
    rows = list(iter_sql_rows(path, 'page', progress=False))
    assert len(rows) == 2
    assert rows[0][0] == 1
    assert rows[0][1] == 0
    assert rows[0][2] == 'Main_Page'
    assert rows[0][3] == 0
    assert rows[1][2] == 'Albert_Einstein'


if __name__ == '__main__':
    import sys
    passed = 0
    failed = 0
    for name, func in list(globals().items()):
        if name.startswith('test_') and callable(func):
            try:
                func()
                print(f"  PASS: {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
