import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import asyncio


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Chunk:
    chunk_id: int
    messages: List[Message]
    token_estimate: int


async def generate_session_title(
        messages: List[Message],
        chunk_target_tokens: int = 8000,
        chunk_overlap_tokens: int = 500,
        section_group_size: int = 10,
) -> Dict[str, Any]:
    """
    返回：
      {
        "chunk_titles": [{"chunk_id":..., "token_estimate":..., "title":...}, ...],
        "final_title": "..."
      }
    """
    chunks = build_chunks(messages, chunk_target_tokens, chunk_overlap_tokens)
    chunk_titles: List[Tuple[int, str]] = []
    records = []

    for c in chunks:
        t = await asyncio.to_thread(title_for_chunk, c)
        chunk_titles.append((c.chunk_id, t))
        records.append({"chunk_id": c.chunk_id, "token_estimate": c.token_estimate, "title": t})

    # 分层合并（避免 chunk 很多时一次合并输入过长）
    titles_level = [t for _, t in chunk_titles]
    while len(titles_level) > section_group_size:
        next_level = []
        for i in range(0, len(titles_level), section_group_size):
            group = titles_level[i: i + section_group_size]
            next_level.append(await asyncio.to_thread(reduce_titles, group))
        titles_level = next_level

    final_title = await asyncio.to_thread(reduce_titles, titles_level) if len(titles_level) > 1 else titles_level[0]

    # 保险：再次截断（防止模型偶尔超字数）
    final_title = final_title.strip()
    if len(final_title) > 20:
        final_title = final_title[:20]

    return {"chunk_titles": records, "final_title": final_title}


def build_chunks(
        messages: List[Message],
        chunk_target_tokens: int = 8000,
        chunk_overlap_tokens: int = 500,
) -> List[Chunk]:
    turns = group_into_turns(messages)

    chunks: List[Chunk] = []
    i = 0
    chunk_id = 1

    while i < len(turns):
        token_sum = 0
        start_i = i
        end_i = i

        while end_i < len(turns):
            t = turns_tokens(turns[end_i])
            if token_sum + t > chunk_target_tokens and end_i > start_i:
                break
            token_sum += t
            end_i += 1

        chunk_turns = turns[start_i:end_i]
        chunk_msgs = [m for turn in chunk_turns for m in turn]
        chunks.append(Chunk(chunk_id=chunk_id, messages=chunk_msgs, token_estimate=token_sum))
        chunk_id += 1

        if end_i >= len(turns):
            break

        # overlap: roll back turns until overlap >= budget
        overlap_tok = 0
        overlap_start = end_i
        j = end_i - 1
        while j >= start_i and overlap_tok < chunk_overlap_tokens:
            overlap_tok += turns_tokens(turns[j])
            overlap_start = j
            j -= 1

        i = overlap_start

    return chunks


def group_into_turns(messages: List[Message]) -> List[List[Message]]:
    turns: List[List[Message]] = []
    cur: List[Message] = []
    for m in messages:
        if m.role == "human":
            if cur:
                turns.append(cur)
            cur = [m]
        else:
            cur.append(m)
    if cur:
        turns.append(cur)
    return turns


def turns_tokens(turn: List[Message]) -> int:
    return sum(estimate_tokens(f"{m.role}: {m.content}\n") for m in turn)


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
    except ImportError:
        tiktoken = None
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - chinese_chars
    return int(chinese_chars * 1.2 + other / 4.0)


def _to_text(messages: List[Message]) -> str:
    return "\n".join([f"{m.role.upper()}: {m.content}" for m in messages])


def title_for_chunk(chunk: Chunk) -> str:
    msgs = [
        {"role": "system", "content": TITLE_ONLY_SYSTEM},
        {"role": "user", "content": f"对话分块（chunk_id={chunk.chunk_id}）：\n\n{_to_text(chunk.messages)}"},
    ]
    return call_llm_glm(msgs)


def reduce_titles(titles: List[str]) -> str:
    msgs = [
        {"role": "system", "content": REDUCE_TITLE_SYSTEM},
        {"role": "user", "content": "分块标题如下（请合并为一个）：\n" + "\n".join(f"- {t}" for t in titles)},
    ]
    return call_llm_glm(msgs)


TITLE_ONLY_SYSTEM = """你是对话标题生成器。
任务：根据输入对话内容，生成一个中文标题，用于会话列表展示。
硬性要求：
- 只输出标题本身，不要输出任何解释、标点外内容、引号、编号、Markdown、JSON
- 中文标题不超过32个汉字
- 英文标题不超过64个字符
- 尽量具体，概括主要话题；不要编造对话里没有的信息
"""

REDUCE_TITLE_SYSTEM = """你是对话标题合并器。
你将收到多个分块标题，请合并为一个更准确的最终中文标题。
硬性要求：
- 只输出标题本身
- 中文标题不超过32个汉字
- 英文标题不超过64个字符
- 不要编造
"""


def call_llm_glm(
        messages: List[Dict[str, str]]
) -> str:
    """
    OpenAI-compatible call.
    Env:
      GLM_API_KEY
      GLM_BASE_URL
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=os.getenv("OLLAMA_API_KEY"),
        base_url="https://ollama.com/v1",
        model="glm-5:cloud",
        temperature=0.2,
        max_tokens=60,
        max_retries=3
    )

    res = llm.invoke(messages)
    return (res.content or "").strip()


if __name__ == "__main__":
    import asyncio as _asyncio

    msgs = [
        Message(role="user", content="你好，想聊聊怎么提高学习效率。"),
        Message(role="assistant", content="可以从目标拆解、番茄钟和复盘开始。"),
        Message(role="user", content="另外解释一下向量数据库是什么。"),
        Message(role="assistant", content="向量数据库用于存储向量并做相似度检索。"),
    ]

    result = _asyncio.run(generate_session_title(msgs))
    print(result["final_title"])
