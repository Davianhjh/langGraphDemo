import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from tools.tool_router import tools


def create_llm():
    """Create and return an LLM instance and a bound version with tools.

    Returns:
        A tuple (llm, llm_with_tools).
    """
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.6-plus:free",
    )
    llm_with_tools = llm.bind_tools(tools)
    return llm, llm_with_tools


class State(TypedDict):
    messages: Annotated[list, add_messages]


def init_graph(llm_with_tools):
    """Initialize and compile the StateGraph with chatbot and tools nodes."""
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
    # tools node
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # 每当调用工具时，我们返回到聊天机器人以决定下一步
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    return graph_builder.compile(checkpointer=checkpointer)


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

    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
        event_message = event["messages"][-1]
        if event_message.type == 'ai':
            print("Assistant:", event_message.content)


def main():
    """Program entrypoint: create LLM, init graph, and run interactive loop."""
    _, llm_with_tools = create_llm()
    # interactive loop (same behavior as original)
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(init_graph(llm_with_tools), user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
