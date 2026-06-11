#!/usr/bin/env bash
# 오프라인 환경에서 npm 패키지 설치
# offline_resources/html2pptx/*.tgz 에서 설치
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(cd "$SCRIPT_DIR/../../offline_resources/html2pptx" && pwd)"

if [ ! -d "$OFFLINE_DIR" ]; then
  echo "ERROR: offline_resources/html2pptx 디렉토리가 없습니다."
  exit 1
fi

TGZ_FILES=("$OFFLINE_DIR"/*.tgz)
if [ ! -f "${TGZ_FILES[0]}" ]; then
  echo "ERROR: offline_resources/html2pptx/*.tgz 파일이 없습니다."
  echo "온라인 환경에서 다음 명령으로 패키지를 준비하세요:"
  echo "  cd deeppresenter/html2pptx && npm pack minimist pptxgenjs playwright"
  exit 1
fi

echo "오프라인 npm 패키지 설치 중..."
cd "$SCRIPT_DIR"
npm install --prefer-offline --no-audit --no-fund \
  "${TGZ_FILES[@]}"

echo "완료: node_modules 설치됨"
