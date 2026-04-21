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

    行为说明：
    - 使用 INSERT IGNORE 批量插入 chat_messages 表（防止重复 message_id 引发错误）
    - 如果插入了新行（affected > 0），会将对应的 chat_dialogs.added_new 设为 1，条件使用 (thread_id, user_id)
    - 函数在失败时尽量保证连接关闭并将异常上抛
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

    if not rows:
        return 0

    with mysql_pool.connection(timeout=5) as conn:
        with conn.cursor() as cur:
            sql1 = "INSERT IGNORE INTO chat_messages (user_id, thread_id, role, content, message_id) VALUES (%s, %s, %s, %s, %s)"
            cur.executemany(sql1, rows)
            affected = cur.rowcount or 0

            # 如果有新插入的消息，将 chat_dialogs.added_new 置为 1，限定 thread_id + user_id
            if affected > 0:
                sql2 = "SELECT id FROM chat_dialogs WHERE thread_id=%s AND user_id=%s for update"
                dialog_id = cur.execute(sql2, (thread_id, user_id))
                if dialog_id:
                    cur.execute("UPDATE chat_dialogs SET added_new=%s WHERE id=%s", (1, dialog_id))
                else:
                    cur.execute("INSERT INTO chat_dialogs (user_id, thread_id, dialog_title, added_new) VALUES (%s, %s, %s, %s)",
                                (user_id, thread_id, messages[0].get("content", "")[:20], 0))
                try:
                    cur.execute("UPDATE chat_dialogs SET added_new=%s WHERE thread_id=%s AND user_id=%s", (1, thread_id, user_id))
                except Exception:
                    # 不要让更新标题的错误阻断消息持久化结果，记录在日志里
                    print(f"Warning: failed to update added_new for thread_id={thread_id} user_id={user_id}")
    return affected


async def summary_chat_messages(dialog_id: int, thread_id: str, messages: List[Message]) -> str:
    if messages:
        try:
            title_res = await generate_session_title(messages)
            new_title = title_res.get("final_title") or ""
            if new_title:
                with mysql_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE chat_dialogs SET dialog_title=%s, added_new=%s WHERE id=%s", (new_title, 0, dialog_id))
                        return new_title
        except Exception as e:
            print(f"Error generating title for dialog_id={dialog_id} thread={thread_id}: {e}")
    return ""
