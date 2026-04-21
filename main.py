from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.chat_bot import init_graph
from db.mysql import mysql_pool

app = FastAPI()
lang_app = init_graph()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    messages: list[ChatMessage]


def sse_data(text: str) -> str:
    # 按 SSE 协议输出一条 data 行
    # 注意要替换换行，否则会被 SSE 当成多条 event
    safe = text.replace("\r", "").replace("\n", "\\n")
    return f"data: {safe}\n\n"


@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id
    user_id = req.user_id

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


@app.get("/history")
async def history(background_tasks: BackgroundTasks, user_id: Optional[str] = None):
    """返回 user_id 对应的会话列表：
    - 如果 chat_dialogs.added_new 为 0，直接返回 id, thread_id, dialog_title
    - 如果 added_new 不为0，查询 chat_messages，生成会话标题并写回 chat_dialogs 后返回
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    dialogs = []
    with mysql_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, thread_id, dialog_title, added_new FROM chat_dialogs WHERE user_id=%s ORDER BY id DESC", (user_id,))
            rows = cur.fetchall()
            for row in rows:
                if 0 == row.get("added_new"):
                    dialogs.append({"id": row["id"], "thread_id": row["thread_id"], "dialog_title": row.get("dialog_title")})
                else:
                    dialog_id = row.get("id")
                    thread_id = row.get("thread_id")
                    # fetch messages for the thread
                    cur.execute("SELECT role, content FROM chat_messages WHERE thread_id=%s ORDER BY id ASC", (thread_id,))
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

    return {"dialogs": dialogs}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
