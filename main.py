from typing import Literal, Optional, List

import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.chat_bot import init_graph
from db.mysql import mysql_pool
from tools.third_party.aliyun_oss_uploader import upload_file as oss_upload_file

app = FastAPI()
lang_app = init_graph()

class ChatFile(BaseModel):
    file_url: str
    file_name: str
    file_ext: str
    mime_type: str
    file_size: int

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    messages: list[ChatMessage]
    files: Optional[List[ChatFile]] = None


# 新增响应模型
class HistoryDialog(BaseModel):
    id: int
    thread_id: str
    dialog_title: str


class HistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    dialogs: List[HistoryDialog]


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    create_time: str


class DialogResponse(BaseModel):
    messages: List[MessageItem]


class DialogListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    messages: List[MessageItem]


def sse_data(text: str) -> str:
    # 按 SSE 协议输出一条 data 行
    # 注意要替换换行，否则会被 SSE 当成多条 event
    safe = text.replace("\r", "").replace("\n", "\\n")
    return f"data: {safe}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id
    user_id = req.user_id
    files = req.files or []

    # 只取本轮最新 user 消消息追加给 LangGraph
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if not last_user:
        # 没有用户消息就直接结束
        return StreamingResponse(iter([sse_data("[DONE]")]), media_type="text/event-stream")

    initial_state = {
        "thread_id": thread_id,
        "user_id": user_id,
        "messages": [
            SystemMessage(
                content="你是一个温暖、准确且有用的助理，能针对用户的各种问题给出答案。并能够判断用户的意图和自己的工具能力匹配时，无论是否缺少参数，优先执行工具调用。"),
            HumanMessage(content=last_user.content)
        ]
    }

    # 关键：把 thread_id 交给 checkpointer，用于恢复/续写同一条对话
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    def event_gen():
        try:
            # updates：每一步仅返回增量字段（避免重复）
            for event in lang_app.stream(initial_state, config=config, stream_mode="updates"):
                for _node_name, update in event.items():
                    if _node_name == "chatbot":
                        new_msg = update.get("messages")[-1]  # 取最新消息
                        if len(new_msg.content) > 0:
                            yield sse_data(new_msg.content)

            yield sse_data("[DONE]")
        except Exception as e:
            print(f"Error in event_gen: {e}")
            # 出错也要结束流，否则前端卡住
            yield sse_data(f"[ERROR] {type(e).__name__}: {e}")
            yield sse_data("[DONE]")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id},
    )


@app.get("/history", response_model=HistoryResponse)
async def history(background_tasks: BackgroundTasks, user_id: Optional[str] = None, page: int = 1, page_size: int = 10):
    """返回 user_id 对应的会话列表（分页）：
    - 如果 chat_dialogs.added_new 为 0，直接返回 id, thread_id, dialog_title
    - 如果 added_new 不为0，查询 chat_messages，生成会话标题并写回 chat_dialogs 后返回

    返回结构：{"total": int, "page": int, "page_size": int, "dialogs": [...]}
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 10:
        page_size = 10

    offset = (page - 1) * page_size

    dialogs = []
    with mysql_pool.connection(timeout=5) as conn:
        with conn.cursor() as cur:
            # total count
            cur.execute("SELECT COUNT(*) as cnt FROM chat_dialogs WHERE user_id=%s", (user_id,))
            total_row = cur.fetchone()
            total = total_row.get("cnt") if total_row else 0

            # fetch paginated dialogs
            cur.execute(
                "SELECT id, thread_id, dialog_title, added_new FROM chat_dialogs WHERE user_id=%s ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                (user_id, page_size, offset))
            rows = cur.fetchall()

            for row in rows:
                if 0 == row.get("added_new"):
                    dialogs.append(
                        {"id": row["id"], "thread_id": row["thread_id"], "dialog_title": row.get("dialog_title")})
                else:
                    dialog_id = row.get("id")
                    thread_id = row.get("thread_id")
                    # fetch messages for the thread
                    cur.execute("SELECT role, content FROM chat_messages WHERE thread_id=%s ORDER BY id ASC",
                                (thread_id,))
                    msgs = cur.fetchall()
                    if msgs:
                        from app.chat_summary import Message
                        from app.chat_message import summary_chat_messages

                        msgs_for_title = []
                        for m in msgs:
                            msgs_for_title.append(Message(role=m.get("role"), content=m.get("content")))
                        background_tasks.add_task(summary_chat_messages, dialog_id, thread_id, msgs_for_title)

                        dialog_title = row.get("dialog_title") or ""
                        dialogs.append({"id": row.get("id"), "thread_id": thread_id, "dialog_title": dialog_title})

                    else:
                        dialogs.append({"id": dialog_id, "thread_id": thread_id, "dialog_title": ""})
        conn.close()
    return {"total": total, "page": page, "page_size": page_size, "dialogs": dialogs}


@app.get("/dialog", response_model=DialogListResponse)
async def dialog(user_id: Optional[str] = None, thread_id: Optional[str] = None, page: int = 1, page_size: int = 10):
    """查询会话的所有消息（按 id 升序）并返回结构化 JSON 列表。

    Query 参数（必填）:
    - user_id: 用户 ID
    - thread_id: 会话 ID

    返回格式:
    {
      "messages": [
        {"id": <int>, "role": <str>, "content": <str>, "create_time": "yyyy-MM-dd HH:mm:ss"},
        ...
      ]
    }

    说明：
    - 如果数据库中的 content/created_at 为 NULL，会分别返回空字符串。
    """
    # 必填校验
    if not user_id or not thread_id:
        raise HTTPException(status_code=400, detail="user_id and thread_id are required")

    # sanitize pagination
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 10:
        page_size = 10

    offset = (page - 1) * page_size

    messages = []
    with mysql_pool.connection(timeout=5) as conn:
        with conn.cursor() as cur:
            # total count
            cur.execute(
                "SELECT COUNT(*) as cnt FROM chat_messages WHERE user_id=%s AND thread_id=%s",
                (user_id, thread_id),
            )
            cnt_row = cur.fetchone()
            total = cnt_row.get("cnt") if cnt_row else 0

            # fetch paginated messages
            cur.execute(
                "SELECT id, role, content, created_at FROM chat_messages WHERE user_id=%s AND thread_id=%s ORDER BY id DESC LIMIT %s OFFSET %s",
                (user_id, thread_id, page_size, offset),
            )
            rows = cur.fetchall()

            for r in rows:
                created_at = r.get("created_at")
                if created_at:
                    try:
                        create_time = created_at.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        # 如果类型不是 datetime，兜底转换为字符串
                        create_time = str(created_at)
                else:
                    create_time = ""

                messages.append({
                    "id": r.get("id"),
                    "role": r.get("role"),
                    "content": r.get("content") or "",
                    "create_time": create_time,
                })
        conn.close()

    return {"total": total, "page": page, "page_size": page_size, "messages": messages}


@app.post("/upload", response_model=ChatFile)
async def upload_file_endpoint(user_id: str = Form(...), file: UploadFile = File(...)):
    """接收 form-data: user_id, file；上传到 Aliyun OSS 并返回结构化信息。

    返回 JSON:
    {
      "file_url": str,
      "file_name": str,
      "file_ext": str,
      "mime_type": str
    }
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read uploaded file: {e}")

    file_size = len(content)

    try:
        url = oss_upload_file(content, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")

    name = file.filename or ""
    base, ext = os.path.splitext(name)
    ext = ext.lstrip('.') if ext else ''

    # Persist file metadata to MySQL (chat_files)
    try:
        with mysql_pool.connection(timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_files (user_id, file_name, file_url, file_size, file_ext, mime_type) VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, name, url, file_size, ext, file.content_type or ""),
                )
                conn.commit()
            conn.close()
        # Note: pool's connections are autocommit=True in config, so no explicit commit needed
    except Exception as e:
        # If DB write fails, return 500 to surface the error (alternatively you could log and continue)
        raise HTTPException(status_code=500, detail=f"failed to persist file metadata: {e}")

    return {
        "file_url": url,
        "file_name": name,
        "file_ext": ext,
        "file_size": file_size,
        "mime_type": file.content_type or "",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
