import os
from typing import Optional, Callable

from langchain_core.tools import tool


@tool(
    description="将 Excel 文件（本地路径或远程 URL）转换为 Markdown，并上传生成的 Markdown 文件。返回上传后的文件 URL 或错误信息。")
def excel_to_markdown(file_path: str) -> str:
    """
    将 Excel 文件转换为 Markdown 的工具（可由模型直接调用）。

    功能:
    - 接受本地文件路径或远程 URL（以 `http://` 或 `https://` 开头）。
    - 若输入为 URL，会下载到临时文件后再转换。
    - 支持单个 sheet 或包含多个 sheet 的 Excel 文件；每个 sheet 会作为一个带标题的段落被转换为 Markdown。
    - 将合并后的 Markdown 文本上传并返回文件的访问 URL。

    参数:
    - `file_path` (str): 本地文件路径或远程文件 URL。

    返回:
    - str: 成功时返回上传后的 Markdown 文件 URL；失败时返回错误描述字符串。

    使用约定:
    - 调用方根据返回值是否为 URL 来判断是否成功（错误信息为可读字符串）。
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

        return process(file_path, upload_func=upload_file)
    except Exception as e:
        print(f"Error converting file to markdown: {str(e)}")
        return f"Failed to convert the file: {str(file_path)}"
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def process(file_path: str, upload_func: Optional[Callable[[bytes, Optional[str]], str]] = None) -> str:
    import pandas as pd
    import time
    import openpyxl

    wb = openpyxl.load_workbook(file_path, data_only=True)
    markdown_content = ""

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        merged_cells_ranges = list(ws.merged_cells.ranges)

        if len(merged_cells_ranges) == 0:
            # --- 原始逻辑（无合并单元格） ---
            data = []
            for row in ws.iter_rows(values_only=True):
                data.append(list(row))

            if not data:
                continue

            df = pd.DataFrame(data)

            if df.shape[0] >= 1:
                raw_header = df.iloc[0].tolist()
                header = [("" if pd.isna(h) else str(h)) for h in raw_header]
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = header
            else:
                continue

            df = df.where(pd.notna(df), "")

            rows_as_str = [[
                ("" if (cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == "") else
                 (cell.decode('utf-8', errors='ignore') if isinstance(cell, (bytes, bytearray)) else str(cell))
                 ) for cell in row]
                for row in df.values.tolist()
            ]
            df = pd.DataFrame(rows_as_str, columns=df.columns)

            sheet_md = f"## Sheet: {sheet_name}\n\n"
            sheet_md += df.to_markdown(index=False)
            markdown_content += sheet_md + "\n\n"
        else:
            # --- HTML 表格逻辑（有合并单元格） ---
            skip_cells = set()
            merge_attrs = {}

            for merged_range in merged_cells_ranges:
                min_col, min_row, max_col, max_row = merged_range.bounds
                colspan = max_col - min_col + 1
                rowspan = max_row - min_row + 1
                merge_attrs[(min_row, min_col)] = {'rowspan': rowspan, 'colspan': colspan}
                for r in range(min_row, max_row + 1):
                    for c in range(min_col, max_col + 1):
                        if r == min_row and c == min_col:
                            continue
                        skip_cells.add((r, c))

            data = []
            for row in ws.iter_rows(values_only=True):
                data.append(list(row))

            if not data:
                continue

            import html

            html_lines = ["<table>"]
            for r_idx, row in enumerate(data, start=1):
                html_lines.append("  <tr>")
                for c_idx, cell_value in enumerate(row, start=1):
                    if (r_idx, c_idx) in skip_cells:
                        continue

                    attrs = ""
                    if (r_idx, c_idx) in merge_attrs:
                        m = merge_attrs[(r_idx, c_idx)]
                        if m['rowspan'] > 1:
                            attrs += f' rowspan="{m["rowspan"]}"'
                        if m['colspan'] > 1:
                            attrs += f' colspan="{m["colspan"]}"'

                    # 安全地转换为字符串
                    if cell_value is None or (isinstance(cell_value, float) and pd.isna(cell_value)):
                        val_str = ""
                    elif isinstance(cell_value, (bytes, bytearray)):
                        val_str = cell_value.decode('utf-8', errors='ignore')
                    else:
                        val_str = str(cell_value)

                    # 转义 HTML 实体，并替换换行符
                    val_str = html.escape(val_str).replace("\n", "<br>")

                    tag = "th" if r_idx == 1 else "td"
                    html_lines.append(f"    <{tag}{attrs}>{val_str}</{tag}>")
                html_lines.append("  </tr>")
            html_lines.append("</table>")

            sheet_md = f"## Sheet: {sheet_name}\n\n"
            sheet_md += "\n".join(html_lines)
            markdown_content += sheet_md + "\n\n"

    # Use a timestamped filename to avoid collisions when uploading multiple times
    filename = f"output_{int(time.time())}.md"
    md_file_url = upload_func(markdown_content.encode("utf-8"), filename)
    return md_file_url
