from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterable, List, Optional, Sequence

import pymysql
from pymysql.cursors import DictCursor

from src.data_infra.config import MySQLConfig


class MySQLClient:
    def __init__(self, config: MySQLConfig) -> None:
        self._config = config

    @contextmanager
    def connection(self) -> Generator[pymysql.connections.Connection, None, None]:
        conn = pymysql.connect(**self._config.dsn_kwargs, cursorclass=DictCursor, autocommit=False)
        with conn.cursor() as cur:
            cur.execute("SET NAMES utf8mb4")
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, sql: str, params: Optional[Sequence] = None) -> int:
        with self.connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, params or ())
            conn.commit()
            return int(affected)

    def executemany(self, sql: str, params_seq: Iterable[Sequence]) -> int:
        rows = list(params_seq)
        if not rows:
            return 0
        with self.connection() as conn:
            with conn.cursor() as cur:
                affected = cur.executemany(sql, rows)
            conn.commit()
            return int(affected)

    def fetchone(self, sql: str, params: Optional[Sequence] = None) -> Optional[dict]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                row = cur.fetchone()
        return row

    def fetchall(self, sql: str, params: Optional[Sequence] = None) -> List[dict]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                rows = cur.fetchall()
        return list(rows)

    def ping(self) -> bool:
        with self.connection() as conn:
            conn.ping(reconnect=True)
        return True
