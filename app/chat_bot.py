import os
import logging
from typing import Annotated, Optional, Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.chat_message import persist_messages_batch
from tools.tool_router import tools, tool_required_args

logger = logging.getLogger(__name__)


class State(TypedDict, total=False):
    user_id: Optional[str]
    thread_id: Optional[str]
    files: Optional[list[dict[str, Any]]]
    messages: Annotated[list, add_messages]
    pending_tool_name: Optional[str]
    pending_tool_args: Optional[dict]
    missing_fields: Optional[list[str]]
    last_persisted_idx: Optional[int]


def _is_arg_missing_or_empty(args: dict, key: str) -> bool:
    """Return True when a required tool argument is absent or effectively empty."""
    if key not in args:
        return True
    value = args[key]
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def create_llm():
    llm = ChatOpenAI(
        api_key=os.getenv("OLLAMA_API_KEY"),
        base_url="https://ollama.com/v1",
        model="minimax-m2.7:cloud",
    )
    return llm.bind_tools(tools)


def decision_node(state: State) -> State:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    files = state.get("files") or []
    file_paths = [f.get("file_url") for f in files if isinstance(f, dict) and f.get("file_url")]
    # 没有工具调用：清理 pending，直接结束
    if not tool_calls:
        return {
            "user_id": state.get("user_id"),
            "thread_id": state.get("thread_id"),
            "files": files,
            "messages": state["messages"],
            "pending_tool_name": None,
            "pending_tool_args": None,
            "missing_fields": None,
            "last_persisted_idx": state.get("last_persisted_idx", -1),
        }

    missing_tool_name = None
    missing_args = None
    missing_fields = None
    file_idx = 0

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        required_args = tool_required_args.get(tool_name, [])
        args = tool_call.get("args")
        if not isinstance(args, dict):
            args = {}
            tool_call["args"] = args

        # 若模型未传 file_path，则按 files 列表顺序自动注入 file_url
        if "file_path" in required_args and _is_arg_missing_or_empty(args, "file_path"):
            if file_idx < len(file_paths):
                args["file_path"] = file_paths[file_idx]
                logger.info("Injected file_path=%s for tool=%s", args["file_path"], tool_name)
                file_idx += 1
            else:
                logger.warning("No remaining file_path to inject for tool=%s", tool_name)

        missing = [k for k in required_args if _is_arg_missing_or_empty(args, k)]
        if missing:
            missing_tool_name = tool_name
            missing_args = args
            missing_fields = missing
            break

    # 缺参：保存 pending，后面交给 request_missing 追问
    if missing_fields:
        return {
            "user_id": state.get("user_id"),
            "thread_id": state.get("thread_id"),
            "files": files,
            "messages": state["messages"],
            "pending_tool_name": missing_tool_name,
            "pending_tool_args": missing_args,
            "missing_fields": missing_fields,
            "last_persisted_idx": state.get("last_persisted_idx", -1),
        }
    else:
        # 参数齐：清掉 pending，让它去 tools node 执行
        return {
            "user_id": state.get("user_id"),
            "thread_id": state.get("thread_id"),
            "files": files,
            "messages": state["messages"],
            "pending_tool_name": None,
            "pending_tool_args": None,
            "missing_fields": None,
            "last_persisted_idx": state.get("last_persisted_idx", -1),
        }


def request_missing_node(state: State) -> State:
    missing = state.get("missing_fields") or []
    parts = [f"为了继续处理，我还需要以下信息："]
    if "file_path" in missing:
        parts.append(f"- 文件路径（file_path）：请提供文件地址/绝对路径")

    return {
        "user_id": state.get("user_id"),
        "thread_id": state.get("thread_id"),
        "files": state.get("files"),
        "messages": [AIMessage(content="\n".join(parts))],
        "pending_tool_args": state.get("pending_tool_args"),
        "pending_tool_name": state.get("pending_tool_name"),
        "missing_fields": missing,
        "last_persisted_idx": state.get("last_persisted_idx", -1),
    }


def route_after_decision(state: State) -> str:
    if state.get("missing_fields"):
        return "request_missing"
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        return "tools"
    return END


def init_graph():
    """Initialize and compile the StateGraph with chatbot and tools nodes."""
    llm_with_tools = create_llm()
    with RedisSaver.from_conn_string(os.getenv("REDIS_URL", "redis://localhost:6379"), ttl={
        "default_ttl": 60,  # Expire checkpoints after 60 minutes
        "refresh_on_read": True,  # Reset expiration time when reading checkpoints
    }) as checkpointer:
        checkpointer.setup()

        def call_llm_with_tools(state: State):
            # state["messages"] 应包含历史消息；传入的最后一条通常是 user 的消息（HumanMessage）
            msgs = state.get("messages") or []
            invoke_msgs = list(msgs)
            files = state.get("files") or []
            if files:
                file_lines = []
                for idx, f in enumerate(files, start=1):
                    if isinstance(f, dict):
                        file_url = f.get("file_url") or ""
                        file_name = f.get("file_name") or ""
                        file_ext = f.get("file_ext") or ""
                        file_lines.append(f"{idx}. file_name={file_name}, file_ext={file_ext}, file_url={file_url}")
                if file_lines:
                    invoke_msgs.append(
                        SystemMessage(
                            content=(
                                "当前可用文件列表如下（转换工具参数 file_path 必须取自 file_url）：\n"
                                + "\n".join(file_lines)
                            )
                        )
                    )

            ai_msg = llm_with_tools.invoke(invoke_msgs)
            # new sequence including the assistant reply
            new_msgs = msgs + [ai_msg]

            # compute the slice to persist
            # Prefer explicit value from state (allow 0); if missing then try to load from checkpointer
            last_idx = state.get("last_persisted_idx", None)
            if last_idx is None:
                # attempt to read from checkpointer using thread_id as key
                last_idx = -1
                thread_id_for_cp = state.get("thread_id")
                if thread_id_for_cp:
                    import json

                    cp_val = None
                    # try common method names that checkpointer implementations might expose
                    for method_name in ("get_checkpoint", "get", "read", "load", "load_checkpoint", "fetch"):
                        fn = getattr(checkpointer, method_name, None)
                        if callable(fn):
                            try:
                                # try calling with thread_id
                                cp_val = fn(thread_id_for_cp)
                            except TypeError:
                                # maybe signature requires no args; try calling without
                                try:
                                    cp_val = fn()
                                except Exception:
                                    cp_val = None
                            except Exception:
                                cp_val = None
                            if cp_val:
                                break

                    # parse returned checkpoint value
                    cp_json = None
                    if isinstance(cp_val, dict):
                        cp_json = cp_val
                    elif isinstance(cp_val, (bytes, str)):
                        try:
                            cp_json = json.loads(cp_val)
                        except Exception:
                            cp_json = None

                    if isinstance(cp_json, dict):
                        # try top-level key, or inside a `state` dict
                        if "last_persisted_idx" in cp_json:
                            last_idx = cp_json.get("last_persisted_idx")
                        elif isinstance(cp_json.get("state"), dict) and "last_persisted_idx" in cp_json.get("state"):
                            last_idx = cp_json.get("state").get("last_persisted_idx")
                        else:
                            # no useful value found; keep default -1
                            last_idx = -1

            start_to_persist = last_idx + 1
            new_last_idx = last_idx

            user_id = state.get("user_id")
            thread_id = state.get("thread_id")
            if thread_id is not None and start_to_persist <= len(new_msgs) - 1:
                to_persist = []
                for msg in new_msgs[start_to_persist:]:
                    if msg.content:
                        to_persist.append(msg)
                try:
                    # batch insert in one transaction
                    persist_messages_batch(user_id, thread_id, to_persist)
                    # treat as success (duplicates are ignored by DB)
                    new_last_idx = len(new_msgs) - 1
                except Exception as e:
                    # on failure, leave last_idx unchanged
                    print(f"Error persisting messages for user={user_id} thread={thread_id}: {e}")
                    new_last_idx = last_idx

            # return assistant message, thread_id and updated last_persisted_idx so state is preserved
            return {
                "user_id": user_id,
                "thread_id": thread_id,
                "files": files,
                "messages": [ai_msg],
                "pending_tool_name": None,
                "pending_tool_args": None,
                "missing_fields": None,
                "last_persisted_idx": new_last_idx,
            }

    graph_builder = StateGraph(State)
    # chatbot node
    graph_builder.add_node("chatbot", call_llm_with_tools)
    # decision node
    graph_builder.add_node("decision", decision_node)
    # request_missing node
    graph_builder.add_node("request_missing", request_missing_node)
    # tools node
    graph_builder.add_node("tools", ToolNode(tools=tools))
    # edges
    graph_builder.add_edge("chatbot", "decision")
    graph_builder.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "request_missing": "request_missing",
            "tools": "tools",
            END: END
        }
    )
    # 每当调用工具时，我们返回到聊天机器人以决定下一步
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("request_missing", END)
    return graph_builder.compile(
        checkpointer=checkpointer
    )


def stream_graph_updates(graph, user_input: str):
    """Stream graph events for a user input and print assistant messages.

    Keeps the original behavior of printing the assistant's last message
    for each event received from graph.stream(...).
    """
    config = {
        "configurable": {
            "thread_id": "1",
            "user_id": "user_123",
        }
    }

    initial_state = {
        "messages": [("system",
                       f"你是一个温暖、准确且有用的助理，能针对用户的各种问题给出答案。并能够判断用户的意图和自己的工具能力匹配时，优先执行工具调用。"),
                      ("user", user_input)],
        # Include thread_id at top-level and inside configurable so nodes can pick it up
        "thread_id": config.get("configurable", {}).get("thread_id") if isinstance(config, dict) else None,
        "user_id": config.get("configurable", {}).get("user_id") if isinstance(config, dict) else None,
    }

    for event in graph.stream(initial_state, config, stream_mode="updates"):
        if "chatbot" in event:
            chatbot_message = event["chatbot"]["messages"][-1]
            if len(chatbot_message.content) > 0:
                print("Assistant:", chatbot_message.content)
        # elif "decision" in event:
        #     print("Graph called decision node.")
        # elif "request_missing" in event:
        #     print("Graph called request_missing node.")
        # elif "tools" in event:
        #     print("Graph called tools node.")


if __name__ == "__main__":
    """Program entrypoint: create LLM, init graph, and run interactive loop."""
    # interactive loop (same behavior as original)
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(init_graph(), user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
