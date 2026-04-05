"""
OSS uploader using environment variables and the `oss2` SDK.

Environment variables (required):
- OSS_ACCESS_KEY_ID
- OSS_ACCESS_KEY_SECRET
- OSS_BUCKET_NAME
- OSS_ENDPOINT   (e.g. 'oss-cn-hangzhou.aliyuncs.com')

Optional:
- OSS_OBJECT_PREFIX (default 'uploads/')

Usage:
from tools.oss_uploader import upload_image
url = upload_image(image_bytes, 'picture.png')

This module delays importing `oss2` until `upload_image` is called so the module can
be imported in environments where `oss2` is not installed (useful for testing).
"""
import os
import uuid
from typing import Optional


def upload_file(image_bytes: bytes, filename: Optional[str] = None) -> str:
    """
    Upload `image_bytes` to Aliyun OSS and return the public URL.

    Reads credentials/config from environment variables. Raises RuntimeError on
    configuration problems, and ImportError if `oss2` is not installed.
    """
    access_key_id = os.getenv('OSS_ACCESS_KEY_ID', os.getenv("OSS_ACCESS_KEY_ID"))
    access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET', os.getenv("OSS_ACCESS_KEY_SECRET"))
    bucket_name = os.getenv('OSS_BUCKET_NAME', 'davian-files')
    endpoint = os.getenv('OSS_ENDPOINT', 'oss-cn-hangzhou.aliyuncs.com')
    prefix = os.getenv('OSS_OBJECT_PREFIX', 'uploads/')

    missing = [name for name, val in (
        ('OSS_ACCESS_KEY_ID', access_key_id),
        ('OSS_ACCESS_KEY_SECRET', access_key_secret),
        ('OSS_BUCKET_NAME', bucket_name),
        ('OSS_ENDPOINT', endpoint),
    ) if not val]
    if missing:
        raise RuntimeError('Missing required OSS environment variables: ' + ', '.join(missing))

    # lazy import so the module can be imported without oss2 installed
    try:
        import oss2
    except Exception as e:
        raise ImportError('oss2 library is required for uploading to OSS. pip install oss2')

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    # Build object name
    ext = ''
    if filename:
        _, ext = os.path.splitext(filename)
    object_basename = uuid.uuid4().hex + (ext or '')
    prefix = prefix or ''
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'
    object_name = prefix + object_basename

    # Perform upload
    # For large objects or advanced use, consider using bucket.put_object_from_file or multipart
    bucket.put_object(object_name, image_bytes)

    # Construct public URL (assuming the bucket is accessible via https://{bucket}.{endpoint}/{object})
    if endpoint.startswith('http://') or endpoint.startswith('https://'):
        # strip scheme if user provided it
        ep = endpoint.split('://', 1)[1]
    else:
        ep = endpoint
    url = f'https://{bucket_name}.{ep}/{object_name}'
    return url


if __name__ == "__main__":
    # Example usage: upload a local file
    with open("D:\\3ee0dafdcfbd4a6d805faea3368616d0.png", 'rb') as f:
        image_bytes = f.read()
    url = upload_file(image_bytes, 'example.png')
    print('Uploaded image URL:', url)
