"""생성된 PPTX 파일을 MinIO 오브젝트 스토리지에 업로드."""

import logging
import os
from functools import lru_cache

from minio import Minio

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> Minio:
    endpoint = os.environ["MINIO_ENDPOINT"]
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


def upload_pptx(local_path: str, emp_no: str, export_filename: str) -> str:
    """local_path의 PPTX 파일을 MinIO에 '{emp_no}/slide/{export_filename}.pptx' 로 업로드.

    반환값은 업로드된 오브젝트 이름(버킷 내 경로).
    """
    bucket = os.environ["MINIO_FILE_BUCKET"]
    filename = export_filename if export_filename.endswith(".pptx") else f"{export_filename}.pptx"
    object_name = f"{emp_no}/slide/{filename}"

    client = _get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    client.fput_object(
        bucket,
        object_name,
        local_path,
        content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    logger.info("[MinIO] uploaded %s -> %s/%s", local_path, bucket, object_name)
    return object_name
