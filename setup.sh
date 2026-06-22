#!/usr/bin/env bash
# 프로젝트 초기 환경 설정 스크립트
# 새 컨테이너/서버에서 처음 한 번만 실행합니다.
#   bash setup.sh
set -e

echo "=== [1/3] 한국어 폰트 설치 (Noto CJK) ==="
apt-get install -y fonts-noto-cjk
fc-cache -fv 2>/dev/null | grep -i "noto\|ko" || true

echo ""
echo "=== [2/3] Python 패키지 설치 ==="
pip install -r requirements.txt

echo ""
echo "=== [3/3] Node.js 패키지 설치 ==="
cd deeppresenter/html2pptx
npm install --no-audit --no-fund
npx playwright install chromium
cd -

echo ""
echo "=== 설정 완료 ==="
