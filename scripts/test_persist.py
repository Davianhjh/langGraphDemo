import os
import sys
from app.chat_message import (
    init_mysql_from_env,
    persist_messages_batch,
    get_last_persisted_idx_redis,
    set_last_persisted_idx_redis,
    add_message,
    get_messages,
)

from langchain_core.messages import HumanMessage, AIMessage


def main():
    thread_id = os.getenv("TEST_THREAD_ID", "test-thread-1")
    print("Using thread_id:", thread_id)

    # Initialize MySQL if env provided
    try:
        init_mysql_from_env()
        print("init_mysql_from_env called")
    except Exception as e:
        print("init_mysql_from_env failed or not configured:", e)

    # Build sample messages
    msgs = [HumanMessage(content="Hello"), AIMessage(content="Hi, how can I help?"), HumanMessage(content="Please search for X"), AIMessage(content="I found results.")]

    # Try batch persist
    try:
        affected = persist_messages_batch(thread_id, msgs)
        print(f"persist_messages_batch affected rows: {affected}")
    except Exception as e:
        print("persist_messages_batch failed:", e)

    # Try setting last persisted idx in redis
    try:
        success = set_last_persisted_idx_redis(thread_id, len(msgs) - 1)
        print("set_last_persisted_idx_redis returned:", success)
        val = get_last_persisted_idx_redis(thread_id)
        print("get_last_persisted_idx_redis returned:", val)
    except Exception as e:
        print("redis checkpoint operations failed:", e)

    # Show in-memory messages
    for m in msgs:
        add_message(thread_id, m)
    print("in-memory messages:", get_messages(thread_id))


if __name__ == '__main__':
    main()
