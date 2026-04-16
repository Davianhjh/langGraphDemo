import io
import os
import time
from typing import Optional, Callable

from langchain_core.tools import tool


@tool(description="将 PDF 转为 Markdown，并上传生成的 Markdown 文件，返回文件 URL 或错误信息")
def pdf_to_markdown(file_path: str) -> str:
    """
    工具：将 PDF 转为 Markdown 并上传（可由模型直接调用）。

    功能概述：
    - 支持传入本地文件路径或远程 URL（以 http:// 或 https:// 开头）；若是远程 URL 会先下载到临时文件再处理。
    - 将 PDF 按页渲染为高质量 PNG（使用 poppler），对每页调用 `GlmOcr` 进行 OCR/结构化解析，汇总为 Markdown 文本。
    - 在 `process` 中可接收一个可选的 `upload_func(bytes, filename) -> str` 用于上传中间图片与最终 Markdown；若存在 `tools.third_party.aliyun_oss_uploader.upload_file` 则默认使用之并返回上传后的文件 URL。

    参数:
        file_path (str): 本地 PDF 文件路径或远程 PDF URL。

    返回:
        str: 成功时返回上传后的 Markdown 文件访问 URL（当上传函数可用并成功时）；否则返回生成的 Markdown 文本。
             出错时返回以 "Failed" 开头的可读错误信息字符串。

    行为与注意事项:
    - 函数为同步接口，适合作为模型工具直接调用：传入 `file_path` 字符串，返回字符串（URL 或错误信息）。
    - 若需结构化返回值或定制上传行为，可直接调用模块内的 `process(input_path, upload_func=...)`。
    - 处理过程中会在 finally 中清理下载的临时文件与渲染中间目录，避免残留。
    - 依赖环境变量 `ZAI_API_KEY`（OCR）和可选的 `POPPLER_PATH`（poppler 可执行路径）。

    示例（伪代码）:
        pdf_to_markdown('C:/tmp/doc.pdf') -> 'https://.../output_123456.md' 或 转换后的 markdown 文本

    模型调用建议:
    - 作为模型工具使用时仅需传入 `file_path`，工具会返回字符串；如需更多元化返回（如 JSON），请在上层封装本工具。
    """
    tmp_file_path = None
    if file_path.startswith("http://") or file_path.startswith("https://"):
        import requests
        import tempfile

        try:
            response = requests.get(file_path)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1]) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            file_path = tmp_file_path
        except Exception as e:
            print(f"Error downloading file from URL: {str(e)}")
            return f"Failed to download the file: {str(file_path)}"

    try:
        from tools.third_party.aliyun_oss_uploader import upload_file

        md = process(file_path, upload_func=upload_file)
        filename = f"output_{int(time.time())}.md"
        md_file_url = upload_file(md.encode("utf-8"), filename)
        return md_file_url
    except Exception as e:
        print(f"Error converting file to markdown: {str(e)}")
        return f"Failed to convert the file: {str(file_path)}"
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def process(input_path: str, upload_func: Optional[Callable[[bytes, Optional[str]], str]] = None) -> str:
    import tempfile
    import uuid

    tmpdir = tempfile.mkdtemp(prefix=f"pdf_out_{uuid.uuid4().hex}_")
    try:
        print("converting pdf to png...")
        converted_image_paths = pdf_to_png_pages(
            pdf_path=input_path,
            out_dir=tmpdir,
            dpi=300,
            poppler_path=os.getenv("POPPLER_PATH"),
            thread_count=2
        )
        print(f"pdf converted to {len(converted_image_paths)} png pages")
        print("ocr png to markdown...")
        markdown_content = ""
        from glmocr import GlmOcr

        for img_path in converted_image_paths:
            with GlmOcr(mode="maas", api_key=os.getenv("ZAI_API_KEY")) as parser:
                parse_result = parser.parse(img_path)
                parsed_markdown_result = parse_result.markdown_result
                parsed_image_files = parse_result.image_files
                if parsed_image_files is not None and len(parsed_image_files) > 0 and upload_func is not None:
                    for img_name, img in parsed_image_files.items():
                        uploaded_image_url = upload_func(image_to_binary(img, "PNG"), img_name)
                        parsed_markdown_result = parsed_markdown_result.replace(f"imgs/{img_name}", uploaded_image_url)

                markdown_content += parsed_markdown_result + "\n\n"
        print(f"pdf to markdown conversion complete")
        return markdown_content
    except Exception as e:
        print(f"Error converting PDF to PNG: {str(e)}")
        raise e
    finally:
        if os.path.exists(tmpdir):
            import shutil
            shutil.rmtree(tmpdir)


def pdf_to_png_pages(
        pdf_path: str,
        out_dir: str,
        dpi: int = 300,
        first_page: int | None = None,
        last_page: int | None = None,
        poppler_path: str | None = None,
        thread_count: int = 4,
) -> list[str]:
    """
    将 PDF 按页渲染为 PNG（无损），以保证质量。
    dpi: 300 通常足够清晰；扫描件或小字可 400-600。
    thread_count: 并行渲染页数（过大可能更慢或占用内存）。
    """
    from pathlib import Path
    from pdf2image import convert_from_path

    os.makedirs(out_dir, exist_ok=True)
    p = Path(pdf_path)
    pdf_name = p.stem

    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt="png",
        first_page=first_page,
        last_page=last_page,
        poppler_path=poppler_path,
        thread_count=thread_count,
        use_pdftocairo=True,  # 更稳的渲染路径，常用于高质量输出
        transparent=False,  # 避免透明背景导致体积变大/边缘异常
        grayscale=False  # 保持彩色（若只要黑白可改 True）
    )

    # 注意：convert_from_path 会按页顺序返回 PIL Images
    converted_image_paths = []
    start = first_page or 1
    for i, img in enumerate(images, start=start):
        out_path = os.path.join(out_dir, f"{pdf_name}_page_{i:04d}.png")
        # PNG 无损，compress_level 越高越小但更慢（0-9）
        img.save(out_path, "PNG", compress_level=6)
        converted_image_paths.append(out_path)

    return converted_image_paths


def image_to_binary(image, image_format):
    # 创建一个内存缓冲区
    buffer = io.BytesIO()
    # 将图片保存到缓冲区，指定格式
    image.save(buffer, image_format)
    # 获取二进制数据
    binary_data = buffer.getvalue()
    # 关闭缓冲区
    buffer.close()
    return binary_data
