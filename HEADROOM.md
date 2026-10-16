# Headroom 연동

[Headroom](https://github.com/headroomlabs-ai/headroom)은 LLM에 보내기 전에 입력을 압축해 토큰을 줄이는 도구다.
이 앱에서는 두 가지 방식으로 쓸 수 있고, **둘 중 하나만** 켠다.

| 방식 | 압축 위치 | 켜는 법 | 절감량 확인 |
|------|----------|--------|------------|
| 라이브러리 모드 | 앱 안 (LLM 호출 직전) | `.env`에 `HEADROOM_ENABLED=true` | `headroom savings` (터미널) |
| 프록시 모드 | 앱과 게이트웨이 사이 프록시 | 프록시 실행 + `LLM_URL`을 프록시로 | 웹 대시보드 |

웹 대시보드는 **프록시를 거친 요청만** 보여준다. 라이브러리 모드의 압축은 대시보드에 나오지 않는다.

## 무엇이 압축되나

두 방식 모두 Headroom의 SmartCrusher가 동작한다. 실제로 줄어드는 것은 **큰 JSON 배열 형태의 도구 결과**다.
같은 모양의 레코드 수백 개가 스키마 한 줄과 CSV 형태의 행으로 바뀐다 (테스트에서 약 70% 절감).
HTML, 마크다운, 코드, 로그는 그대로 전달된다. 그래서 문서마다 절감량이 0일 수도 있다.

## 1. 라이브러리 모드

```bash
pip install -r requirements.txt   # headroom-ai 포함
```

`.env`:

```
HEADROOM_ENABLED=true
# HEADROOM_MIN_TOKENS=500                  # 이보다 작은 도구 결과는 건드리지 않음
# HEADROOM_EXCLUDE_TOOLS=read_file,finalize # 절대 압축하지 않을 도구
```

- 압축은 모델에 보내는 메시지에만 적용된다. 에이전트 대화 기록과 `.history/` 원본은 그대로다.
- `read_file` 결과는 기본으로 제외한다. 모델이 읽은 내용을 그대로 `edit_file`에 복사해 쓰기 때문이다.
- 서버 시작 시 `Headroom compression: on` 로그가 나오면 켜진 것이다. 첫 import에 몇 초가 걸려서 시작할 때 미리 불러온다.
- 호출별 절감량은 `.history/*-llm-calls.json`의 `headroom_est_tokens_saved`, 합계는 `total_headroom_est_tokens_saved`에 남는다.
- 사내망용으로 Headroom의 외부 접속(통계 전송, 업데이트 확인, 모델 다운로드)은 꺼둔다.

누적 절감량 보기:

```bash
headroom savings
```

`Savings by client` 항목에 `ppt-agent`로 표시된다. 기록은 `~/.headroom/savings_events.jsonl`에 쌓인다.

## 2. 프록시 모드 + 대시보드

프록시 기능은 추가 패키지가 필요하다 (transformers 등, 수백 MB).

```bash
pip install "headroom-ai[proxy]"
```

**터미널 1 — 프록시 실행** (인자는 실제 LLM 게이트웨이 주소, 생략하면 SK hynix 게이트웨이):

```bash
bash headroom-proxy.sh http://workplace-litellm.aipp02.skhynix.com/v1
```

**`.env` 수정** — 앱이 프록시로 보내게 하고, 라이브러리 모드는 끈다 (이중 압축 방지):

```
LLM_URL=http://127.0.0.1:8787/v1
HEADROOM_ENABLED=false
```

**터미널 2 — 앱 실행:**

```bash
python main-ui.py
```

**대시보드:** 브라우저에서 <http://127.0.0.1:8787/dashboard> 를 연다.
WSL2에서는 Windows 브라우저로 같은 주소를 열면 된다. 안 열리면 `headroom-proxy.sh`의 `headroom proxy` 명령에 `--host 0.0.0.0`을 추가하고 WSL IP로 접속한다.

`Session` 탭은 프록시를 켠 뒤의 요청, `Lifetime` 탭은 누적값이다. 비용 절감액은 모델 가격 정보가 없으면 $0으로 나온다. 토큰 절감량은 정상으로 집계된다.

### headroom-proxy.sh 가 켜는 옵션과 이유

| 옵션 | 이유 |
|------|------|
| `--no-ccr` | 기본값은 요청에 `headroom_retrieve` 도구를 몰래 추가한다. 이 앱은 그 도구를 실행할 수 없어 모델이 호출하면 에이전트가 에러가 난다. |
| `--no-cache` | 같은 요청에 캐시된 응답을 돌려주지 않는다. 에이전트 재시도가 같은 답만 받는 것을 막는다. |
| `--no-rate-limit` | 기본 분당 60회 제한을 끈다. 병렬 Design 워커가 넘길 수 있다. |
| `HEADROOM_OFFLINE=1` | Headroom이 스스로 여는 외부 연결(익명 통계 전송, 업데이트 확인, 모델 다운로드)을 모두 막는다. 기본값으로는 익명 세션 통계를 외부로 보낸다. |
| `LITELLM_LOCAL_MODEL_COST_MAP=True` | litellm이 GitHub에서 가격표를 받으려다 실패하면 요청마다 몇 초씩 멈춘다. 내장 가격표를 쓰게 한다. |

API 키는 앱이 보낸 그대로 게이트웨이에 전달된다. 프록시에 따로 키를 넣을 필요는 없다.

## 끄기

- 라이브러리 모드: `HEADROOM_ENABLED=false` (기본값)
- 프록시 모드: `LLM_URL`을 원래 게이트웨이 주소로 되돌리고 프록시를 끈다.
