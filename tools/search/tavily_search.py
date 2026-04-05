import os

from langchain_core.tools import tool


@tool(description="使用 Tavily 搜索工具进行网络搜索，返回前 10 个结果的标题和链接。输入为搜索查询字符串，输出为格式化的搜索结果列表。")
def tavily_search(query: str) -> str:
    """
    Tavily 网络搜索工具（模型可直接调用）。

    简介:
    - 这是一个供自动化代理/模型调用的搜索工具，用于使用 Tavily API 执行网络搜索并返回前 N 条结果的标题和链接。
    - 函数由 @tool 装饰器注册为可被 LLM 工具调用。

    输入 (query: str):
    - query: 要搜索的查询字符串，非空。

    输出 (str):
    - 返回一个字符串，包含按行编号的搜索结果，每行格式为:
      "1. 标题 - 链接"
    - 如果未找到结果，返回 "No results found."。

    JSON 示例（供模型参考）:
    {
      "tool": "tavily_search",
      "input": {
        "query": "Python web scraping 教程"
      }
    }

    错误处理:
    - 当环境变量 TAVILY_API_KEY 未设置或调用失败时，函数会抛出异常（由上层代理捕获或记录）。

    注意:
    - 该工具默认返回前 10 条结果（max_results=10）。如需更改，请在实现中调整 max_results 参数或扩展为可配置参数。
    """

    from langchain_tavily import TavilySearch

    search_tool = TavilySearch(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_results=10
    )
    return search_tool.invoke(query)