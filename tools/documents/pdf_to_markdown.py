import os
from typing import Optional, Callable


def pdf_to_markdown(file_path: str) -> str:
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

        return process(file_path, upload_func=upload_file)
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
        print(f"pdf converted to png pages in {tmpdir}")
        print("ocr png to markdown...")

    except Exception as e:
        print(f"Error converting PDF to PNG: {str(e)}")
        return f"Failed to convert PDF to PNG: {str(input_path)}"
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


if __name__ == "__main__":
    process("C:\\Users\\hujinhua\\Downloads\\612791ee9086649533592aec215cb720.pdf")
