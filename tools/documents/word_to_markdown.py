import base64
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from typing import Callable, Optional, Tuple, List

from langchain_core.tools import tool


@tool(description="将 Word 文档或远程 Word 文件转换为 Markdown，并上传生成的 Markdown 文件。"
                  "输入为本地文件路径或可访问的 URL，返回 Markdown 文本与产物 URL 列表；失败时返回错误信息与空列表。")
def word_to_markdown(file_path: str) -> str:
    """
    将 Word 文档转换为 Markdown 的工具（可由模型直接调用）。

    功能:
    - 接受本地文件路径或远程 URL（以 `http://` 或 `https://` 开头）。
    - 若输入为 URL，会下载到临时文件后再转换。
    - 使用内部转换函数生成 Markdown，并将结果上传以返回文件 URL。

    参数:
    - `file_path` (str): 本地文件路径或远程文件 URL。

    返回:
    - Tuple[str, List[str]]:
      - 第一项: 成功时为生成的 Markdown 文本；失败时为错误描述字符串。
      - 第二项: 产物 URL 列表（成功时包含上传后的 Markdown 文件 URL，失败时为空列表）。

    使用约定:
    - 调用方应根据第一项的类型/内容判断是否成功（错误信息以字符串形式返回）。
    """
    from tools.third_party.aliyun_oss_uploader import upload_file

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
        md = process(file_path, upload_func=upload_file)
        md_file_url = upload_file(md.encode("utf-8"), "output.md")
        return md_file_url
    except Exception as e:
        print(f"Error converting file to markdown: {str(e)}")
        return f"Failed to convert the file: {str(file_path)}"
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def process(input_path: str, upload_func: Optional[Callable[[bytes, Optional[str]], str]] = None) -> str:
    """
    High-level convenience: take a .doc or .docx path and return a Markdown string.
    """
    # prepare docx path
    ext = os.path.splitext(input_path)[1].lower()
    cleanup_tmp = False
    if ext == '.doc':
        print("converting doc to docx...")
        docx_path = convert_doc_to_docx(input_path)
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
            td = os.path.dirname(docx_path)
            try:
                shutil.rmtree(td)
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


def convert_doc_to_docx(input_path: str) -> str:
    """
    If input_path is a .docx, return it unchanged. If it's a .doc, attempt to convert
    to a temporary .docx file and return that path.

    Conversion methods (in order):
    - win32com (requires pywin32, only on Windows with MS Word installed)
    - soffice (LibreOffice / OpenOffice) commandline conversion

    Raises RuntimeError if conversion fails.
    """
    input_path = os.path.abspath(input_path)
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.docx':
        return input_path
    if ext != '.doc':
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
