# TODO: 사내 registry의 실제 Python 베이스 이미지로 교체
FROM dockerhub.hynix.com/python:3.11-slim

ENV user_dir=/root
ENV work_dir=/app

COPY ./ $work_dir/
WORKDIR $work_dir

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -U websockets

# TODO: 실제 준비한 zip 파일명으로 교체.
# apt로 unzip 패키지를 따로 설치할 필요 없게 파이썬 내장 zipfile 모듈로 압축을 품.
# (chmod +x는 zipfile 모듈이 실행 권한 비트를 못 살릴 수도 있어서 넣는 안전장치)
COPY chrome-linux64.zip /tmp/
RUN mkdir -p $work_dir/offline_resources/ms-playwright/chromium-1228 \
    && python -m zipfile -e /tmp/chrome-linux64.zip $work_dir/offline_resources/ms-playwright/chromium-1228/ \
    && chmod +x $work_dir/offline_resources/ms-playwright/chromium-1228/chrome-linux64/chrome \
    && rm /tmp/chrome-linux64.zip \
    && ls -la $work_dir/offline_resources/ms-playwright/chromium-1228/chrome-linux64

EXPOSE 5000
