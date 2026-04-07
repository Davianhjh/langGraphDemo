from tools.documents.word_to_markdown import word_to_markdown
from tools.documents.excel_to_markdown import excel_to_markdown
from tools.search.tavily_search import tavily_search

tools = [tavily_search, word_to_markdown, excel_to_markdown]


