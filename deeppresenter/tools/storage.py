"""생성된 PPTX 파일을 MinIO 오브젝트 스토리지에 업로드/다운로드."""

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

PPTX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@lru_cache(maxsize=1)
def _get_client() -> Minio:
    endpoint = os.environ["MINIO_ENDPOINT"]
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


def _object_name(emp_no: str, export_filename: str, suffix: str = "") -> str:
    filename = export_filename if export_filename.endswith(".pptx") else f"{export_filename}.pptx"
    if suffix:
        filename = f"{Path(filename).stem}{suffix}.pptx"
    return f"{emp_no}/slide/{filename}"


def _object_exists(client: Minio, bucket: str, object_name: str) -> bool:
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchBucket", "NoSuchObject"):
            return False
        raise


def _resolve_unique_object_name(client: Minio, bucket: str, emp_no: str, export_filename: str) -> str:
    """{emp_no}/slide/{export_filename}.pptx 가 이미 있으면 '_(1)', '_(2)', ... 를 붙여 비어있는 이름을 찾는다."""
    object_name = _object_name(emp_no, export_filename)
    if not _object_exists(client, bucket, object_name):
        return object_name

    counter = 1
    while True:
        candidate = _object_name(emp_no, export_filename, suffix=f"_({counter})")
        if not _object_exists(client, bucket, candidate):
            return candidate
        counter += 1


def upload_pptx(local_path: str, emp_no: str, export_filename: str) -> str:
    """local_path의 PPTX 파일을 MinIO에 '{emp_no}/slide/{export_filename}.pptx' 로 업로드.

    같은 경로에 동일 파일명이 이미 있으면 '{export_filename}_(1).pptx', '_(2).pptx', ... 순으로
    비어있는 이름을 찾아 저장한다 (기존 파일을 덮어쓰지 않음).

    반환값은 업로드된 오브젝트 이름(버킷 내 경로).
    """
    bucket = os.environ["MINIO_FILE_BUCKET"]
    client = _get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    object_name = _resolve_unique_object_name(client, bucket, emp_no, export_filename)
    client.fput_object(bucket, object_name, local_path, content_type=PPTX_CONTENT_TYPE)
    logger.info("[MinIO] uploaded %s -> %s/%s", local_path, bucket, object_name)
    return object_name


def download_pptx(emp_no: str, export_filename: str) -> tuple[str, str]:
    """MinIO의 '{emp_no}/slide/{export_filename}.pptx' 오브젝트를 임시 파일로 내려받는다.

    반환값은 (내려받은 로컬 파일 경로, 오브젝트 이름). 존재하지 않으면 FileNotFoundError.
    """
    bucket = os.environ["MINIO_FILE_BUCKET"]
    object_name = _object_name(emp_no, export_filename)

    fd, tmp_path = tempfile.mkstemp(suffix=".pptx")
    os.close(fd)

    client = _get_client()
    try:
        client.fget_object(bucket, object_name, tmp_path)
    except S3Error as e:
        Path(tmp_path).unlink(missing_ok=True)
        if e.code in ("NoSuchKey", "NoSuchBucket"):
            raise FileNotFoundError(f"Object not found: {bucket}/{object_name}") from e
        raise

    logger.info("[MinIO] downloaded %s/%s -> %s", bucket, object_name, tmp_path)
    return tmp_path, object_name
