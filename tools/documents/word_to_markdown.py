import base64
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from typing import Callable, Optional

from langchain_core.tools import tool


@tool(description="将 Word (.doc/.docx/.rtf) 转为 Markdown，并上传生成的 Markdown 文件，返回文件 URL 或错误信息")
def word_to_markdown(file_path: str) -> str:
    """
    工具：将 Word 文档转换为 Markdown 并上传（可由模型直接调用）。

    功能概述：
    - 支持输入类型：`.doc`, `.docx`, `.rtf`；同时支持以 `http://` 或 `https://` 开头的远程 URL（会先下载到临时文件）。
    - 若为 `.doc` 或 `.rtf`，会先调用 LibreOffice（soffice）将其转换为 `.docx`，然后使用 mammoth 将 `.docx` 转为 HTML，再使用 pypandoc 将 HTML 转为 GitHub-flavored Markdown（GFM）。
    - 在将 Word 转为 HTML 时，图片会通过 `upload_func` 上传（默认使用内置的 data-uri 上传器），并将图片链接内嵌到最终的 Markdown 中。
    - 返回值为上传并保存后的 Markdown 文件 URL（字符串）。若发生错误，则返回可读的错误信息字符串。

    参数：
    - `file_path` (str): 本地文件路径或远程文件 URL（http/https）。

    返回：
    - str: 成功时返回上传后的 Markdown 文件访问 URL；失败时返回错误描述字符串。

    错误处理与注意：
    - 函数会在内部调用 `tools.third_party.aliyun_oss_uploader.upload_file` 上传生成的 Markdown 文件以及转换过程中的图片（若提供 `upload_func` 则使用之）。
    - 若你在本地测试或不希望真实上传，可调用模块内的 `process(input_path, upload_func=your_stub)` 来注入一个本地上传函数 `upload_func(bytes, filename) -> str`，以便把图片或 Markdown 写到本地并获得本地路径作为返回值。
    - 若系统缺少依赖（mammoth 或 pypandoc），`process` 内会抛出 RuntimeError，调用方应根据返回信息辨识失败原因。

    模型调用建议：
    - 模型作为工具调用时只需传入 `file_path` 字符串，工具会同步返回字符串：成功时为 URL，失败时为错误信息。
    - 若需要结构化返回（例如 JSON 包含 success/data/error），建议在上层对本工具进行封装或请求我把返回改为 JSON。

    示例（伪代码）：
        word_to_markdown('C:/tmp/report.docx') -> 'https://.../output_123456.md'

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
    # prepare docx path
    ext = os.path.splitext(input_path)[1].lower()
    cleanup_tmp = False
    if ext == '.doc' or ext == '.rtf':
        print("converting doc to docx...")
        docx_path = convert_to_docx(input_path)
        cleanup_tmp = True
    elif ext == '.docx':
        docx_path = input_path
    else:
        raise ValueError('Unsupported input type: ' + ext)

    try:
        print("converting docx to html...")
        html = docx_to_html(docx_path, upload_func=upload_func)
        print("converting html to markdown...")
        md = html_to_markdown(html)
        print("word to markdown conversion complete")
        return md
    finally:
        if cleanup_tmp:
            # delete the temp dir containing the docx
            try:
                os.remove(docx_path)
            except Exception:
                pass


# Optional deps
try:
    import mammoth
except Exception as e:
    mammoth = None

try:
    import pypandoc
except Exception as e:
    pypandoc = None


def _ensure_deps_available():
    missing = []
    if mammoth is None:
        missing.append('mammoth')
    if pypandoc is None:
        missing.append('pypandoc')
    if missing:
        raise RuntimeError(
            f"Missing python packages: {', '.join(missing)}. Add them to requirements.txt and pip install.")


def convert_to_docx(input_path: str) -> str:
    input_path = os.path.abspath(input_path)
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.docx':
        return input_path
    if ext != '.doc' and ext != '.rtf':
        raise ValueError('Unsupported file extension: ' + ext)

    # create a temporary output path
    tmp_dir = tempfile.mkdtemp(prefix='docx_conv_')

    # Try soffice (LibreOffice) conversion
    try:
        soffice = shutil.which("soffice") or shutil.which("soffice.exe")
        if not soffice:
            raise RuntimeError("soffice not found. Add LibreOffice\\program to PATH or use full path.")
        # soffice may output into the cwd; run with --outdir
        cmd = [soffice, '--headless', '--convert-to', 'docx', '--outdir', str(tmp_dir), str(input_path)]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        # generated filename
        generated = os.path.join(tmp_dir, os.path.splitext(os.path.basename(input_path))[0] + '.docx')
        if os.path.exists(generated):
            return generated
        else:
            raise RuntimeError('soffice conversion did not produce expected file')
    except Exception as e:
        # cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError('Failed to convert .doc to .docx: ' + str(e))


def docx_to_html(docx_path: str, upload_func: Optional[Callable[[bytes, Optional[str]], str]] = None) -> str:
    _ensure_deps_available()
    try:
        # Prefer mammoth's image inline converter wrapper when present
        images_api = getattr(mammoth, 'images', None)
        if images_api is not None and hasattr(images_api, 'inline'):
            convert_image = images_api.inline(_make_mammoth_image_converter(upload_func))
            with open(docx_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file, convert_image=convert_image)
        else:
            with open(docx_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
        html = result.value
        return html
    except Exception as e:
        raise RuntimeError('Failed to convert docx to HTML: ' + str(e))


def html_to_markdown(html: str) -> str:
    _ensure_deps_available()
    md_text = pypandoc.convert_text(
        html,
        to="gfm",
        format="html",
    )
    return md_text


def _default_upload_func(image_bytes: bytes, filename: Optional[str]) -> str:
    """
    Default upload function: return a data URI so the markdown contains an inline image.
    This is used when the caller doesn't provide their own upload function.
    """
    content_type, _ = mimetypes.guess_type(filename or '')
    if content_type is None:
        content_type = 'application/octet-stream'
    b64 = base64.b64encode(image_bytes).decode('ascii')
    return f'data:{content_type};base64,{b64}'


def _make_mammoth_image_converter(upload_func: Optional[Callable[[bytes, Optional[str]], str]]):
    """
    Returns a function suitable for mammoth's convert_image option. The converter will
    call `upload_func(image_bytes, filename)` and return { 'src': url }.
    """
    uploader = upload_func or _default_upload_func

    def convert_image(image):
        try:
            with image.open() as f:
                image_bytes = f.read()

            if not image_bytes:
                raise ValueError(
                    f"Empty image stream (content_type={getattr(image, 'content_type', None)})"
                )

            # determine filename (mammoth may provide content_type)
            extension = None
            try:
                content_type = getattr(image, 'content_type', None)
                if content_type:
                    extension = mimetypes.guess_extension(content_type) or ''
            except Exception:
                extension = ''
            suggested_name = f"{int(time.time())}_{uuid.uuid4().hex}{extension or ''}"
            url = uploader(image_bytes, suggested_name)
            return {"src": url}

        except Exception as e:
            print("mammoth image handler failed:", repr(e))
            print("content_type:", getattr(image, "content_type", None))
            traceback.print_exc()
            return {"src": ""}

    return convert_image
