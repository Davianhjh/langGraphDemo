import io
import os
import time
from typing import Optional, Callable

from glmocr import GlmOcr


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
        print("pdf to markdown conversion complete")
        return markdown_content
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


if __name__ == "__main__":
    md_file = pdf_to_markdown("C:\\Users\\hujinhua\\Downloads\\学科网资料2024072901(9)份\\第1讲 测量和机械运动（集训本）-【学海风暴·PK中考】2024中考物理备考（江西专用）\\第1讲 测量和机械运动.pdf")
    print(md_file)
