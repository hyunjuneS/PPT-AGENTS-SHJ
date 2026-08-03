"""생성된 PPTX/HTML 슬라이드를 MinIO 오브젝트 스토리지에 업로드/다운로드.

한 번의 생성 요청이 만든 산출물은 전부 '{emp_no}/{export_filename stem}/' 아래
카테고리별 하위 폴더에 모인다:
  - ppt/            : PPTX 파일 1개
  - htmls/          : 슬라이드 html N개 + global.css + 로컬 이미지 (개별 파일)
  - combined_html/  : 합쳐진 스크롤 html + 그 로컬 이미지 (개별 파일)
  - history/        : .history/ 안의 {agent}-history.json / {agent}-config.json (실행 성공/실패 무관하게 업로드됨)
"""

import logging
import mimetypes
import os
import re
import shutil
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

PPTX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
HTML_CONTENT_TYPE = "text/html"
CSS_CONTENT_TYPE = "text/css"


@lru_cache(maxsize=1)
def _get_client() -> Minio:
    endpoint = os.environ["MINIO_ENDPOINT"]
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


def _filename_stem(export_filename: str) -> str:
    """export_filename에서 .pptx 확장자를 뗀 stem을 반환 — 이 stem이 '{emp_no}/{stem}/' 폴더명이자
    ppt/htmls/combined_html 세 카테고리가 공유하는 파일/폴더명의 기준이 된다."""
    return export_filename[:-len(".pptx")] if export_filename.endswith(".pptx") else export_filename


def _next_stem(stem: str) -> str:
    """'{stem}' -> '{stem}_(1)', '{stem}_(1)' -> '{stem}_(2)', ..."""
    m = re.match(r"^(.*)_\((\d+)\)$", stem)
    if m:
        return f"{m.group(1)}_({int(m.group(2)) + 1})"
    return f"{stem}_(1)"


def _object_exists(client: Minio, bucket: str, object_name: str) -> bool:
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchBucket", "NoSuchObject"):
            return False
        raise


def _content_type_for(path: Path) -> str:
    if path.suffix == ".css":
        return CSS_CONTENT_TYPE
    if path.suffix in (".html", ".htm"):
        return HTML_CONTENT_TYPE
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _object_name(emp_no: str, export_filename: str, suffix: str = "") -> str:
    stem = _filename_stem(export_filename)
    if suffix:
        stem = f"{stem}{suffix}"
    return f"{emp_no}/{stem}/ppt/{stem}.pptx"


def _resolve_unique_object_name(client: Minio, bucket: str, emp_no: str, export_filename: str) -> str:
    """{emp_no}/{stem}/ppt/{stem}.pptx 가 이미 있으면 '_(1)', '_(2)', ... 를 붙여 비어있는 이름을 찾는다."""
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
    """local_path의 PPTX 파일을 MinIO에 '{emp_no}/{export_filename stem}/ppt/{export_filename stem}.pptx' 로 업로드.

    같은 경로에 동일 파일명이 이미 있으면 '_(1)', '_(2)', ... 순으로 비어있는 이름을 찾아
    저장한다 (기존 파일을 덮어쓰지 않음).

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


def _upload_files_under_prefix(local_files: list[str], emp_no: str, export_filename: str, category: str) -> list[str]:
    """local_files를 MinIO에 '{emp_no}/{export_filename stem}/{category}/{원본 파일명}' 으로
    각각 개별 업로드. 해당 stem 폴더가 이미 있으면(첫 파일 존재 여부로 판단) '_(1)', '_(2)', ...
    를 붙여 비어있는 폴더를 찾는다 (기존 파일을 덮어쓰지 않음). 반환값은 업로드된 오브젝트 이름 목록.
    """
    if not local_files:
        return []

    bucket = os.environ["MINIO_FILE_BUCKET"]
    client = _get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    files = [Path(f) for f in local_files]
    stem = _filename_stem(export_filename)
    while _object_exists(client, bucket, f"{emp_no}/{stem}/{category}/{files[0].name}"):
        stem = _next_stem(stem)

    object_names = []
    for f in files:
        object_name = f"{emp_no}/{stem}/{category}/{f.name}"
        client.fput_object(bucket, object_name, str(f), content_type=_content_type_for(f))
        object_names.append(object_name)

    logger.info("[MinIO] uploaded %d file(s) -> %s/%s/%s/%s/", len(object_names), bucket, emp_no, stem, category)
    return object_names


def upload_html_files(local_files: list[str], emp_no: str, export_filename: str) -> list[str]:
    """슬라이드 html N개 + global.css + 로컬 이미지(local_files)를 MinIO에
    '{emp_no}/{export_filename stem}/htmls/{원본 파일명}' 으로 각각 개별 업로드.
    반환값은 업로드된 오브젝트 이름 목록."""
    return _upload_files_under_prefix(local_files, emp_no, export_filename, category="htmls")


def upload_combined_html(local_files: list[str], emp_no: str, export_filename: str) -> list[str]:
    """세로로 스크롤 가능하게 합쳐진 combined.html + 그 안에서 상대경로로 참조하는 global.css/
    로컬 이미지(local_files)를 MinIO에 '{emp_no}/{export_filename stem}/combined_html/{원본 파일명}'
    으로 각각 개별 업로드 — combined.html 안 각 슬라이드는 여전히 <link href="global.css">로
    스타일을 가져오고 이미지도 상대경로 그대로이므로, global.css와 이미지들이 combined.html과
    같은 폴더에 함께 있어야 정상적으로 렌더링된다. 반환값은 업로드된 오브젝트 이름 목록."""
    return _upload_files_under_prefix(local_files, emp_no, export_filename, category="combined_html")


def upload_history_files(local_files: list[str], emp_no: str, export_filename: str) -> list[str]:
    """.history/ 안의 {agent}-history.json / {agent}-config.json 을 MinIO에
    '{emp_no}/{export_filename stem}/history/{원본 파일명}' 으로 업로드.

    ppt/htmls/combined_html과 달리 한 번의 요청 안에서도 여러 번 불릴 수 있다 (Research 단계가
    끝나자마자 한 번, 이어지는 Design 단계가 끝난 뒤 다시 한 번 — 혹은 어느 한쪽이 실패해도 그
    시점까지의 내용으로 한 번) — 그때마다 최신 상태를 반영해야 하는 디버깅용 아티팩트이므로,
    다른 카테고리처럼 '_(1)', '_(2)' 버전 폴더를 만들지 않고 같은 경로에 그대로 덮어쓴다.
    """
    if not local_files:
        return []

    bucket = os.environ["MINIO_FILE_BUCKET"]
    client = _get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    stem = _filename_stem(export_filename)
    object_names = []
    for f in (Path(p) for p in local_files):
        object_name = f"{emp_no}/{stem}/history/{f.name}"
        client.fput_object(bucket, object_name, str(f), content_type=_content_type_for(f))
        object_names.append(object_name)

    logger.info("[MinIO] uploaded %d history file(s) -> %s/%s/%s/history/", len(object_names), bucket, emp_no, stem)
    return object_names


def download_pptx(emp_no: str, export_filename: str) -> tuple[str, str]:
    """MinIO의 '{emp_no}/{export_filename stem}/ppt/{export_filename stem}.pptx' 오브젝트를
    임시 파일로 내려받는다.

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


def _download_files_under_prefix(emp_no: str, export_filename: str, category: str) -> tuple[str, str]:
    """MinIO의 '{emp_no}/{export_filename stem}/{category}/' 아래 파일 전부를 하나의 zip으로
    묶어 임시 파일로 반환한다. 반환값은 (내려받은 zip 로컬 경로, 오브젝트 prefix).
    그 prefix 아래 파일이 하나도 없으면 FileNotFoundError."""
    bucket = os.environ["MINIO_FILE_BUCKET"]
    stem = _filename_stem(export_filename)
    prefix = f"{emp_no}/{stem}/{category}/"

    client = _get_client()
    object_names = [obj.object_name for obj in client.list_objects(bucket, prefix=prefix, recursive=True)]
    if not object_names:
        raise FileNotFoundError(f"No objects found under: {bucket}/{prefix}")

    work_dir = Path(tempfile.mkdtemp())
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(zip_fd)

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for object_name in object_names:
                arcname = object_name[len(prefix):]
                local_path = work_dir / arcname
                client.fget_object(bucket, object_name, str(local_path))
                zf.write(local_path, arcname=arcname)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("[MinIO] downloaded %d file(s) from %s/%s -> %s", len(object_names), bucket, prefix, zip_path)
    return zip_path, prefix


def download_html_files(emp_no: str, export_filename: str) -> tuple[str, str]:
    """MinIO의 '{emp_no}/{export_filename stem}/htmls/' 아래 개별 슬라이드 html + css + 이미지
    파일을 모두 받아 하나의 zip으로 묶어 임시 파일로 반환한다."""
    return _download_files_under_prefix(emp_no, export_filename, category="htmls")


def download_combined_html(emp_no: str, export_filename: str) -> tuple[str, str]:
    """MinIO의 '{emp_no}/{export_filename stem}/combined_html/' 아래 combined.html + 그 로컬
    이미지 파일을 모두 받아 하나의 zip으로 묶어 임시 파일로 반환한다."""
    return _download_files_under_prefix(emp_no, export_filename, category="combined_html")
