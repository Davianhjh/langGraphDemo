import time
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage

# MySQL persistence configuration (populated by init_mysql_from_env)
_db_config: Dict[str, Any] = {}


def _now_ts() -> float:
    return time.time()


def _message_to_record(message: Any) -> Dict[str, Any]:
    """
    将传入的 message 转为统一的字典记录。
    支持：AIMessage, HumanMessage, 或者直接传入 str（视为 HumanMessage 内容）。
    返回字段：{ "role": "ai"|"human", "content": str, "timestamp": float, "raw": original }
    """
    id = getattr(message, "id", None)
    if isinstance(message, AIMessage):
        role = "ai"
        content = getattr(message, "content", "")
    elif isinstance(message, HumanMessage):
        role = "human"
        content = getattr(message, "content", "")
    elif isinstance(message, str):
        role = "human"
        content = message
    else:
        # 尝试按常见属性读取
        role = getattr(message, "role", None) or getattr(message, "type", None)
        if role is None:
            # 默认当作 human
            role = "human"
        content = getattr(message, "content", str(message)) if hasattr(message, "content") else str(message)

    return {
        "id": id,
        "role": role,
        "content": content,
        "timestamp": _now_ts(),
        "raw": message,
    }


# ----------------- MySQL persistence helpers -----------------
def init_mysql_from_env(prefix: str = "MYSQL_") -> None:
    """
    从环境变量读取 MySQL 配置并确保消息表存在。

    支持的环境变量：MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
    """
    import os
    global _db_config
    host = os.getenv(prefix + "HOST", "localhost")
    port = int(os.getenv(prefix + "PORT", "3306"))
    user = os.getenv(prefix + "USER", "root")
    password = os.getenv(prefix + "PASSWORD", "")
    db = os.getenv(prefix + "DB", "langgraph")

    _db_config = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "db": db,
        "charset": "utf8mb4",
    }


def _get_conn():
    """建立并返回一个 pymysql 连接（按需导入 pymysql）。"""
    import pymysql
    cfg = _db_config
    if not cfg:
        raise RuntimeError("MySQL is not initialized.")
    return pymysql.connect(
        host=cfg.get("host"),
        port=cfg.get("port", 3306),
        user=cfg.get("user"),
        password=cfg.get("password"),
        database=cfg.get("db"),
        charset=cfg.get("charset", "utf8mb4"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
        connect_timeout=5,
        read_timeout=30,
        write_timeout=30,
    )


def persist_messages_batch(thread_id: str, messages: List[Any]) -> int:
    """
    在单个数据库事务中批量持久化 messages 列表（按顺序），返回实际插入的行数。
    """
    if not thread_id:
        raise ValueError("thread_id is required to persist messages")
    if not messages:
        return 0

    rows = []
    for m in messages:
        rec = _message_to_record(m)
        role = rec["role"]
        if 'ai' == role or 'human' == role:
            content = rec.get("content")
            message_id = rec.get("id")
            rows.append((thread_id, role, content, message_id))

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = "INSERT IGNORE INTO chat_messages (thread_id, role, content, message_id) VALUES (%s, %s, %s, %s)"
            cur.executemany(sql, rows)
            affected = cur.rowcount or 0
        conn.commit()
        return affected
    finally:
        conn.close()
