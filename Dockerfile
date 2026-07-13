# TODO: 사내 registry의 실제 Python 베이스 이미지로 교체 (예: dockerhub.hynix.com/python:3.11-slim)
FROM dockerhub.hynix.com/python:3.11-slim

ENV user_dir=/root
ENV work_dir=/app

COPY .pip.conf $user_dir/.pip/pip.conf

COPY ./ $work_dir/
WORKDIR $work_dir

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -U websockets

# TODO: 사내 apt 미러 주소로 교체 (폐쇄망이라 공식 저장소 접근 불가)
RUN printf "deb [사내 apt 미러 URL] stable main\n" > /etc/apt/sources.list.d/internal.list

# fonts-noto-cjk: 한국어 슬라이드를 Chromium으로 렌더링할 때 한글이 깨지지 않도록 필수.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg xz-utils fonts-noto-cjk \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# TODO: 실제 준비한 Node.js tar.xz 파일명으로 교체.
# 버전은 코드에 고정되어 있지 않음(package.json에 engines 제약 없음) — Node 18+ LTS면 충분.
COPY node-v20.20.0-linux-x64.tar.xz /tmp/
RUN tar -xJf /tmp/node-v20.20.0-linux-x64.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node-v20.20.0-linux-x64.tar.xz \
    && node --version && npm --version

# deeppresenter/tools/export.py가 PPTX 변환을 위해 이 node_modules를 서브프로세스로 실행함.
RUN cd $work_dir/deeppresenter/html2pptx && \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install --omit=dev --no-audit --no-fund && \
    node -e "require('playwright'); require('pptxgenjs'); require('fast-glob'); require('minimist'); require('sharp'); console.log('deps ok')"

# Chromium은 폐쇄망이라 zip으로 직접 올려서 압축 해제.
# 실행 파일 탐색은 PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH 환경변수(.env)로 지정하므로
# 압축 해제 위치 자체는 자유롭게 잡아도 무방 (deeppresenter/tools/export.py 참고).
# TODO: 실제 준비한 zip 파일명으로 교체.
COPY chrome-linux64.zip /tmp/
RUN mkdir -p $work_dir/offline_resources/ms-playwright/chromium-1228 \
    && unzip -o /tmp/chrome-linux64.zip -d $work_dir/offline_resources/ms-playwright/chromium-1228/ \
    && rm /tmp/chrome-linux64.zip \
    && ls -la $work_dir/offline_resources/ms-playwright/chromium-1228/chrome-linux64

EXPOSE 5000

ENTRYPOINT ["python", "main-ui.py"]
