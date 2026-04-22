import time
import hashlib
import json
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
    first_human_message = ""
    first_human_files_written = False
    for m in messages:
        rec = _message_to_record(m)
        role = rec["role"]
        if 'ai' == role or 'human' == role:
            content = rec.get("content")
            message_id = rec.get("id")
            files_json = None
            if role == "human" and not first_human_files_written:
                raw = rec.get("raw")
                msg_files = None
                if hasattr(raw, "additional_kwargs"):
                    msg_files = (getattr(raw, "additional_kwargs", {}) or {}).get("files")
                if msg_files is not None:
                    try:
                        files_json = json.dumps(msg_files, ensure_ascii=False)
                        first_human_files_written = True
                    except Exception:
                        files_json = None

            rows.append((user_id, thread_id, role, content, message_id, files_json))
            if 'human' == role and not first_human_message:
                first_human_message = content

    if not rows:
        return 0

    with mysql_pool.connection(timeout=5) as conn:
        with conn.cursor() as cur:
            sql1 = "INSERT IGNORE INTO chat_messages (user_id, thread_id, role, content, message_id, files) VALUES (%s, %s, %s, %s, %s, %s)"
            cur.executemany(sql1, rows)
            conn.commit()
            affected = cur.rowcount or 0

            # 如果有新插入的消息，将 chat_dialogs.added_new 置为 1，限定 thread_id + user_id
            if affected > 0:
                sql2 = "SELECT id FROM chat_dialogs WHERE thread_id=%s AND user_id=%s for update"
                dialog_id = cur.execute(sql2, (thread_id, user_id))
                if not dialog_id:
                    cur.execute(
                        "INSERT INTO chat_dialogs (user_id, thread_id, dialog_title, added_new) VALUES (%s, %s, %s, %s)",
                        (user_id, thread_id, first_human_message[:64], 0))
                try:
                    cur.execute("UPDATE chat_dialogs SET added_new=%s WHERE thread_id=%s AND user_id=%s",
                                (1, thread_id, user_id))
                except Exception:
                    # 不要让更新标题的错误阻断消息持久化结果，记录在日志里
                    print(f"Warning: failed to update added_new for thread_id={thread_id} user_id={user_id}")
                conn.commit()
        conn.close()
    return affected


async def summary_chat_messages(dialog_id: int, thread_id: str, messages: List[Message]) -> str:
    if not messages:
        return ""

    lock_name = "summary_lock_" + hashlib.sha1(thread_id.encode("utf-8")).hexdigest()
    new_title = ""
    with mysql_pool.connection(timeout=5) as conn:
        with conn.cursor() as cur:
            # 立即尝试获取命名锁（timeout=0），如果获取失败说明已有其他进程在处理，直接返回
            cur.execute("SELECT GET_LOCK(%s, %s) as got", (lock_name, 0))
            row = cur.fetchone()
            got = row.get("got") if row else 0
            if got != 1:
                # 未获取到锁，说明已有并发任务在处理该 thread_id，跳过
                print(f"summary already running for thread_id={thread_id}, skip")
            else:
                # 成功获取锁，开始生成会话标题
                try:
                    print(f"start generating title for dialog_id={dialog_id} thread={thread_id} with {len(messages)} messages")
                    title_res = await generate_session_title(messages)
                    new_title = title_res.get("final_title") or ""
                    if new_title:
                        cur.execute("UPDATE chat_dialogs SET dialog_title=%s, added_new=%s WHERE id=%s",
                                    (new_title, 0, dialog_id))
                        conn.commit()
                except Exception as e:
                    print(f"Error generating title for dialog_id={dialog_id} thread={thread_id}: {e}")
        conn.close()
    return new_title
