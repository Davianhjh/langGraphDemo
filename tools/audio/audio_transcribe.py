import os

from langchain_core.tools import tool
from openai import OpenAI


@tool(description="将本地音频文件或远程音频 URL 转写为中文文本；本地文件会先上传到阿里云 OSS（upload_file），最后返回转写文本或错误信息。")
def audio_transcribe(file_path: str) -> str:
    """
    工具：将音频文件转写为中文文本（可由模型直接调用）。

    功能概述：
    - 接受本地文件路径或远程 URL（以 http:// 或 https:// 开头）。
    - 如果传入本地路径，函数会把文件读取为 bytes 并调用 `tools.third_party.aliyun_oss_uploader.upload_file` 上传；
      upload_file 需要返回可公开访问的 URL 字符串，函数随后会使用该 URL 调用 ASR 模型。
    - 如果传入的是远程 URL，函数会直接把该 URL 当作音频位置传给下游 `process`。
    - 不对音频做本地转码或解码，假定上传或提供的音频格式由下游模型支持。

    参数：
    - file_path (str): 本地绝对路径或远程音频 URL（http(s)）。

    返回：
    - str: 成功时返回模型转写的文本；失败时返回包含错误信息的字符串（不会抛出异常）。

    使用说明（模型作为工具调用）：
    - 直接调用本工具并传入 `file_path`（本地路径或 URL）。
    - 工具保证：无异常抛出；出错时返回可读的错误字符串，便于模型或调用方判断失败原因。

    注意：若需要结构化返回（例如 JSON），可请求我修改返回格式为 JSON 字符串。
    """
    try:
        from tools.third_party.aliyun_oss_uploader import upload_file

        if not file_path.startswith("http://") and not file_path.startswith("https://"):
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                file_path = upload_file(file_bytes, os.path.basename(file_path))

        return process(file_path)
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return f"Failed to transcribe the audio: {str(file_path)}"


def process(file_path: str) -> str:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # qwen3-asr-flash 模型适用于语音转文本任务，支持多种语言和方言，具有较高的准确率和实时性。后续可以换本地部署的qwen3-asr模型。
    completion = client.chat.completions.create(
        model="qwen3-asr-flash",
        messages=[
            {
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": file_path
                        }
                    }
                ],
                "role": "user"
            }
        ],
        stream=False,
        # stream设为False时，不能设置stream_options参数
        # stream_options={"include_usage": True},
        extra_body={
            "asr_options": {
                # "language": "zh",
                "enable_itn": False
            }
        }
    )
    content = completion.choices[0].message.content
    return content
