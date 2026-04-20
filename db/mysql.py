from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Iterator, Optional, Type

import pymysql


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"

    # pymysql connect options
    cursorclass: Optional[Type] = pymysql.cursors.DictCursor
    autocommit: bool = True
    connect_timeout: int = 10
    read_timeout: int = 30
    write_timeout: int = 30


class PyMySQLPool:
    """
    A simple, thread-safe PyMySQL connection pool.

    - Borrow:  conn = pool.get_conn()
    - Return:  pool.release_conn(conn)
    - Preferred: with pool.connection() as conn: ...
    """

    def __init__(self, config: MySQLConfig, pool_size: int = 10):
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")

        self._config = config
        self._pool_size = pool_size
        self._queue: Queue[pymysql.connections.Connection] = Queue(maxsize=pool_size)
        self._init_lock = threading.Lock()
        self._initialized = False

        self._initialize()

    def _initialize(self) -> None:
        # Avoid double init in weird import scenarios
        with self._init_lock:
            if self._initialized:
                return
            for _ in range(self._pool_size):
                self._queue.put(self._new_conn())
            self._initialized = True

    def _new_conn(self) -> pymysql.connections.Connection:
        return pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            charset=self._config.charset,
            cursorclass=self._config.cursorclass,
            autocommit=self._config.autocommit,
            connect_timeout=self._config.connect_timeout,
            read_timeout=self._config.read_timeout,
            write_timeout=self._config.write_timeout,
        )

    def _ensure_alive(self, conn: pymysql.connections.Connection) -> pymysql.connections.Connection:
        """
        Ensure the connection is alive before giving it to caller.
        - ping(reconnect=True) will reconnect if dropped.
        """
        try:
            conn.ping(reconnect=True)
            return conn
        except Exception:
            # If ping fails badly, rebuild the connection.
            try:
                conn.close()
            except Exception:
                pass
            return self._new_conn()

    def get_conn(self, timeout: Optional[float] = None) -> pymysql.connections.Connection:
        """
        Borrow a connection from pool. If timeout is None, block forever.
        """
        conn = self._queue.get(block=True, timeout=timeout)  # may raise queue.Empty
        return self._ensure_alive(conn)

    def release_conn(self, conn: pymysql.connections.Connection) -> None:
        """
        Return a connection back to pool.
        If connection is closed, replace it with a new one to keep pool size stable.
        """
        try:
            # PyMySQL has .open boolean for connection state
            if not getattr(conn, "open", True):
                conn = self._new_conn()
        except Exception:
            conn = self._new_conn()

        # Never block forever on put; if queue is full, just close it.
        try:
            self._queue.put(conn, block=False)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    @contextmanager
    def connection(self, timeout: Optional[float] = None) -> Iterator[pymysql.connections.Connection]:
        """
        Context manager: auto return connection to pool.
        """
        conn = self.get_conn(timeout=timeout)
        try:
            yield conn
        finally:
            self.release_conn(conn)

    def closeall(self) -> None:
        """
        Close all pooled connections (best-effort).
        """
        while True:
            try:
                conn = self._queue.get(block=False)
            except Empty:
                break
            try:
                conn.close()
            except Exception:
                pass


cfg = MySQLConfig(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_NAME", "langgraph"),
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

mysql_pool = PyMySQLPool(cfg, pool_size=10)
