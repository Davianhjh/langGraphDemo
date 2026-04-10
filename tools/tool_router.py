from tools.documents.word_to_markdown import word_to_markdown
from tools.documents.excel_to_markdown import excel_to_markdown
from tools.audio.audio_transcribe import audio_transcribe
from tools.search.tavily_search import tavily_search

tools = [
    tavily_search,
    word_to_markdown,
    audio_transcribe,
    excel_to_markdown
]

tool_required_args = {
    "tavily_search": ["query"],
    "word_to_markdown": ["file_path"],
    "audio_transcribe": ["file_path"],
    "excel_to_markdown": ["file_path"],
}


