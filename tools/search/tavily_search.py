import os

from langchain_core.tools import tool


@tool(description="在 Tavily 上执行搜索并返回结果字符串。接受查询文本，使用环境变量 TAVILY_API_KEY 进行鉴权。")
def tavily_search(query: str) -> str:
    """
    在 Tavily 上执行搜索工具（可由模型直接调用）。

    功能:
    - 接受自然语言查询字符串。
    - 使用环境变量 `TAVILY_API_KEY` 作为 API key。
    - 返回 TavilySearch.invoke 的原始字符串结果（取决于第三方库的返回格式）。

    参数:
    - `query` (str): 要搜索的文本查询。

    返回:
    - str: 成功时返回搜索结果字符串；失败时返回错误描述字符串（便于模型作为工具消费）。

    错误处理:
    - 若未设置 `TAVILY_API_KEY`，返回可读错误信息而不是抛出异常。
    - 捕获并返回调用过程中的异常消息，避免抛出未捕获异常给调用者。
    """
    try:
        from langchain_tavily import TavilySearch

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "错误: 未设置环境变量 TAVILY_API_KEY"

        search_tool = TavilySearch(
            tavily_api_key=api_key,
            max_results=10
        )
        return search_tool.invoke(query)
    except Exception as e:
        # 返回错误字符串，便于作为工具调用时模型理解失败原因
        return f"搜索调用失败: {str(e)}"

