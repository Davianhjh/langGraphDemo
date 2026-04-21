import time
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage

from app.chat_summary import Message, generate_session_title
from db.mysql import mysql_pool


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


def persist_messages_batch(user_id: str, thread_id: str, messages: List[Any]) -> int:
    """
    在单个数据库事务中批量持久化 messages 列表（按顺序），返回实际插入的行数。
    """
    if not user_id:
        raise ValueError("user_id is required to persist messages")
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
            rows.append((user_id, thread_id, role, content, message_id))

    try:
        with mysql_pool.connection(timeout=5) as conn:
            with conn.cursor() as cur:
                sql = "INSERT IGNORE INTO chat_messages (user_id, thread_id, role, content, message_id) VALUES (%s, %s, %s, %s, %s)"
                cur.executemany(sql, rows)
                affected = cur.rowcount or 0
        conn.commit()
        return affected
    finally:
        conn.close()


async def summary_chat_messages(dialog_id: int, thread_id: str, messages: List[Message]):
    if messages:
        try:
            title_res = await generate_session_title(messages)
            new_title = title_res.get("final_title") or ""
            if new_title:
                with mysql_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE chat_dialogs SET dialog_title=%s, added_new=%s WHERE id=%s", (new_title, 0, dialog_id))
        except Exception as e:
            print(f"Error generating title for dialog_id={dialog_id} thread={thread_id}: {e}")
