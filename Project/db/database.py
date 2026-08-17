"""
数据库操作层 - 封装 SQLite 连接和常用 CRUD 操作。
所有对话记录、文档注册、会话管理均通过此类持久化。
"""

import sqlite3
import os
import threading
from datetime import datetime
from typing import Optional

import config


# 线程局部存储：每个线程持有独立的 SQLite 连接
_thread_local = threading.local()


class Database:
    """SQLite 数据库连接管理类（线程安全）。

    每个线程使用独立的 sqlite3.Connection，避免
    "SQLite objects created in a thread can only be used in that same thread" 错误。
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接。每个线程一个连接。"""
        if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            _thread_local.conn = conn
        return _thread_local.conn

    def _init_db(self):
        """读取 schema.sql 并执行建表语句。"""
        conn = self._get_conn()
        schema_path = config.SCHEMA_SQL
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                sql = f.read()
            conn.executescript(sql)
        else:
            # 如果 schema.sql 不存在，内联建表
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT CHECK(role IN ('user', 'assistant', 'system')) NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT DEFAULT 'qa',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id, timestamp);
                CREATE TABLE IF NOT EXISTS document_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    filepath TEXT NOT NULL,
                    chunks_count INTEGER DEFAULT 0,
                    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                    title TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_id ON sessions(session_id);
            """)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """执行一条 SQL 语句并返回游标。"""
        conn = self._get_conn()
        return conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        """批量执行 SQL 语句。"""
        conn = self._get_conn()
        return conn.executemany(sql, params_list)

    def commit(self):
        """提交当前事务。"""
        conn = self._get_conn()
        conn.commit()

    def close(self):
        """关闭当前线程的数据库连接。"""
        if hasattr(_thread_local, "conn") and _thread_local.conn:
            _thread_local.conn.close()
            _thread_local.conn = None

    # ---- 对话记录操作 ----

    def save_message(self, session_id: str, role: str, content: str,
                     source: str = "qa") -> int:
        """保存一条对话消息，返回新记录的 id。"""
        cur = self.execute(
            "INSERT INTO conversations (session_id, role, content, source) VALUES (?, ?, ?, ?)",
            (session_id, role, content, source)
        )
        self.commit()
        return cur.lastrowid

    def save_messages_batch(self, session_id: str, messages: list):
        """
        批量保存消息。
        messages: [(role, content, source), ...]
        """
        rows = [(session_id, role, content, source) for role, content, source in messages]
        self.executemany(
            "INSERT INTO conversations (session_id, role, content, source) VALUES (?, ?, ?, ?)",
            rows
        )
        self.commit()

    def get_history(self, session_id: str, last_n: int = 20) -> list:
        """获取指定会话最近 N 条消息，按时间正序排列。"""
        cur = self.execute(
            """SELECT role, content, source, timestamp
               FROM conversations
               WHERE session_id = ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (session_id, last_n)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_all_messages(self, session_id: str) -> list:
        """获取指定会话的全部消息。"""
        cur = self.execute(
            """SELECT role, content, source, timestamp
               FROM conversations
               WHERE session_id = ?
               ORDER BY timestamp ASC""",
            (session_id,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_all_sessions(self) -> list:
        """列出所有有对话记录的会话 ID。"""
        cur = self.execute(
            """SELECT DISTINCT session_id, MAX(timestamp) as last_active
               FROM conversations
               GROUP BY session_id
               ORDER BY last_active DESC"""
        )
        return [dict(row) for row in cur.fetchall()]

    def get_session_count(self) -> int:
        """获取会话总数。"""
        cur = self.execute("SELECT COUNT(DISTINCT session_id) as cnt FROM conversations")
        return cur.fetchone()["cnt"]

    def get_source_stats(self, session_id: str) -> dict:
        """统计指定会话中各来源的消息数量。"""
        cur = self.execute(
            """SELECT source, COUNT(*) as cnt
               FROM conversations
               WHERE session_id = ?
               GROUP BY source""",
            (session_id,)
        )
        return {row["source"]: row["cnt"] for row in cur.fetchall()}

    # ---- 会话管理 ----

    def create_session(self, session_id: str, title: str = "") -> bool:
        """创建新会话记录。"""
        try:
            self.execute(
                "INSERT INTO sessions (session_id, title) VALUES (?, ?)",
                (session_id, title)
            )
            self.commit()
            return True
        except sqlite3.IntegrityError:
            # 会话已存在，更新时间戳
            self.execute(
                "UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            self.commit()
            return False

    def delete_session(self, session_id: str) -> int:
        """删除指定会话的所有消息，返回删除的记录数。"""
        self.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        self.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.commit()
        return self.execute(
            "SELECT changes()"
        ).fetchone()[0]

    # ---- 文档注册 ----

    def register_document(self, filename: str, filepath: str, chunks_count: int = 0) -> bool:
        """注册已导入的文档。"""
        try:
            self.execute(
                "INSERT INTO document_registry (filename, filepath, chunks_count) VALUES (?, ?, ?)",
                (filename, filepath, chunks_count)
            )
            self.commit()
            return True
        except sqlite3.IntegrityError:
            self.execute(
                "UPDATE document_registry SET chunks_count = ? WHERE filename = ?",
                (chunks_count, filename)
            )
            self.commit()
            return False

    def list_documents(self) -> list:
        """列出所有已注册的文档。"""
        cur = self.execute(
            "SELECT filename, filepath, chunks_count, ingested_at FROM document_registry ORDER BY ingested_at DESC"
        )
        return [dict(row) for row in cur.fetchall()]

    def get_total_chunks(self) -> int:
        """获取知识库中文档片段的总数。"""
        cur = self.execute("SELECT COALESCE(SUM(chunks_count), 0) as total FROM document_registry")
        return cur.fetchone()["total"]

    def clear_all(self):
        """清空所有数据（调试用）。"""
        self.execute("DELETE FROM conversations")
        self.execute("DELETE FROM document_registry")
        self.execute("DELETE FROM sessions")
        self.commit()
