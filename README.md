# wordToMarkDown

Small utility to convert Word `.docx` and `.doc` files to Markdown. Images inside the
Word file are extracted and passed to a user-provided upload function. If no upload
function is provided the images are embedded as data URIs.

Usage (Python API):

```python
from tools.documents.word_to_markdown import word_to_markdown

# Optionally implement an upload function with signature: (bytes, filename) -> url
# def upload_to_oss(image_bytes, filename):
#     # upload to Aliyun OSS and return public URL
#     return 'https://example.com/' + filename

md = word_to_markdown('example.docx', upload_func=None)
print(md)
```

Usage (CLI):

python tools\wordToMarkDown.py --input path/to/file.docx --out out.md
python tools\wordToMarkDown.py --input doc.doc --upload-module tools.oss_uploader:upload_image

OSS upload implementation (environment variables)

A ready-to-use uploader is provided at `tools/oss_uploader.py`. It reads the following
environment variables (required):

- OSS_ACCESS_KEY_ID
- OSS_ACCESS_KEY_SECRET
- OSS_BUCKET_NAME
- OSS_ENDPOINT   (e.g. 'oss-cn-hangzhou.aliyuncs.com')

Optional:
- OSS_OBJECT_PREFIX (default 'uploads/')

Example usage:

```powershell
# Set env vars in PowerShell
$env:OSS_ACCESS_KEY_ID = 'your-key-id'
$env:OSS_ACCESS_KEY_SECRET = 'your-key-secret'
$env:OSS_BUCKET_NAME = 'your-bucket'
$env:OSS_ENDPOINT = 'oss-cn-hangzhou.aliyuncs.com'

python .\tools\wordToMarkDown.py --input .\example.docx --upload-module tools.oss_uploader:upload_image --out out.md
```

Notes:
- For `.doc` files the script will try to use pywin32 (Windows MS Word) or LibreOffice
  (`soffice`) to convert to `.docx` before processing.
- You must install dependencies from `requirements.txt`.
- The upload function should accept `(bytes, filename)` and return a URL string.
