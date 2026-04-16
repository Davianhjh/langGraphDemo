import os
from typing import Annotated, Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools.tool_router import tools, tool_required_args


class State(TypedDict):
    messages: Annotated[list, add_messages]
    pending_tool_name: Optional[str]
    pending_tool_args: Optional[dict]
    missing_fields: Optional[list[str]]


def create_llm():
    llm = ChatOpenAI(
        api_key=os.getenv("OLLAMA_API_KEY"),
        base_url="https://ollama.com/v1",
        model="gemma4:31b-cloud",
    )
    return llm.bind_tools(tools)


def decision_node(state: State) -> State:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    # 没有工具调用：清理 pending，直接结束
    if not tool_calls:
        return {
            "messages": state["messages"],
            "pending_tool_name": None,
            "pending_tool_args": None,
            "missing_fields": None,
        }

    tool_call = tool_calls[0]  # 目前假设每次只调用一个工具
    tool_name = tool_call["name"]
    args = tool_call.get("args", {}) or {}

    required_args = tool_required_args.get(tool_name, [])
    missing = [k for k in required_args if k not in args]

    # 缺参：保存 pending，后面交给 request_missing 追问
    if missing:
        return {
            "messages": state["messages"],
            "pending_tool_name": tool_name,
            "pending_tool_args": args,
            "missing_fields": missing,
        }
    else:
        # 参数齐：清掉 pending，让它去 tools node 执行
        return {
            "messages": state["messages"],
            "pending_tool_name": None,
            "pending_tool_args": None,
            "missing_fields": None,
        }


def request_missing_node(state: State) -> State:
    missing = state.get("missing_fields") or []
    parts = [f"为了继续处理，我还需要以下信息："]
    if "file_path" in missing:
        parts.append(f"- 文件路径（file_path）：请提供文件地址/绝对路径")

    return {
        "messages": [AIMessage(content="\n".join(parts))],
        "pending_tool_args": state.get("pending_tool_args"),
        "pending_tool_name": state.get("pending_tool_name"),
        "missing_fields": missing,
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
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

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
            "thread_id": "1"
        }
    }

    for event in graph.stream({"messages": [("system",
                                             f"你是一个温暖、准确且有用的助理，能针对用户的各种问题给出答案。并能够判断用户的意图和自己的工具能力匹配时，优先执行工具调用。"),
                                            ("user", user_input)]}, config, stream_mode="updates"):
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
