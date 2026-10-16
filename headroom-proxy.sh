#!/usr/bin/env bash
# Headroom 프록시 + 대시보드 실행 스크립트 (HEADROOM.md 참고)
#
#   bash headroom-proxy.sh [upstream_url]
#
# upstream_url: 프록시가 요청을 넘겨줄 실제 LLM 게이트웨이 (기본: SK hynix LiteLLM 게이트웨이).
# .env 의 LLM_URL 은 이 프록시(http://127.0.0.1:8787/v1)를 가리키게 바꾸고, HEADROOM_ENABLED=false 로 둔다.
# 대시보드: http://127.0.0.1:${HEADROOM_PORT:-8787}/dashboard
set -euo pipefail

UPSTREAM="${1:-http://workplace-litellm.aipp02.skhynix.com/v1}"
PORT="${HEADROOM_PORT:-8787}"

# 사내망 설정 — 외부로 나가는 연결(익명 통계 전송, 업데이트 확인, 모델 다운로드,
# litellm 가격표 다운로드)을 모두 끈다. 가격표 다운로드가 막히면 요청마다 수 초씩 멈춘다.
export HEADROOM_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True

echo "Headroom proxy  : http://127.0.0.1:${PORT}/v1  ->  ${UPSTREAM}"
echo "Dashboard       : http://127.0.0.1:${PORT}/dashboard"

# --no-ccr        : 압축 본문에 복원 마커를 넣지 않고 headroom_retrieve 도구도 주입하지 않는다.
#                   이 앱은 그 도구를 실행할 수 없어서, 모델이 호출하면 에이전트가 에러가 난다.
# --no-cache      : 같은 요청에 캐시된 응답을 돌려주지 않는다 (에이전트 재시도가 같은 답만 받는 것 방지).
# --no-rate-limit : 기본 분당 60회 제한을 끈다 (병렬 Design 워커가 넘길 수 있음).
exec headroom proxy \
  --port "${PORT}" \
  --openai-api-url "${UPSTREAM}" \
  --no-ccr \
  --no-cache \
  --no-rate-limit
