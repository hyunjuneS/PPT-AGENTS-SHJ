"""Kompress(Headroom의 AI 기반 텍스트 압축) 단독 테스트.

앱에 붙이기 전에, 원고(.md)를 Kompress로 압축해 보고 두 가지를 확인한다.
  1. 얼마나 줄어드는가 (토큰 수, 절감률, 걸린 시간)
  2. 무엇이 빠지는가 (원문에 있던 숫자 중 압축본에서 사라진 것, 많이 지워진 단어)

사용법 (저장소 루트에서):
    python test/kompress_test.py                       # sample.md 테스트
    python test/kompress_test.py 원고1.md 원고2.md      # 여러 파일
    python test/kompress_test.py 원고.md --target-ratio 0.5   # 더 세게 (약 50%만 남김)
    python test/kompress_test.py 원고.md --online       # 모델을 인터넷에서 직접 받기 (사내망에서는 보통 막힘)

결과는 test/kompress_output/ 에 저장된다.
    <이름>.kompressed.md   압축된 원고
    <이름>.report.md       절감량, 사라진 숫자, 많이 지워진 단어

모델 파일이 필요하다 (HEADROOM.md 의 Kompress 항목 참고). 없으면 받는 방법을 출력하고 멈춘다.
주의: Kompress 의 바탕인 ModernBERT 는 영어 위주로 학습된 모델이라, 한국어 원고는
압축률이 낮거나 중요한 단어가 지워질 수 있다. 한국어 원고로 꼭 확인할 것.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "sample.md"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "kompress_output"

# headroom 이 고정해서 쓰는 모델 버전 (headroom-ai 0.41.0 기준). 이 버전 그대로 받아야 한다.
MODEL_REPOS = [
    ("chopratejas/kompress-v2-base", "b1563631b35bfdcee37587ad530147497d820d4c", "압축 모델 (ONNX)"),
    ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8", "토크나이저"),
]

# 숫자, 소수, 퍼센트, 날짜(2024-03-01, 24.10.15), 시간 등을 한 덩어리로 잡는다.
_NUMBER_RE = re.compile(r"\d+(?:[.,:/\-]\d+)*%?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kompress 단독 압축 테스트")
    p.add_argument("files", nargs="*", type=Path, default=[DEFAULT_INPUT], help="테스트할 .md/.txt 파일 (기본: sample.md)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="결과 저장 폴더")
    p.add_argument("--target-ratio", type=float, default=None,
                   help="남길 비율 (예: 0.5 = 약 50%%만 남김). 생략하면 Kompress 가 스스로 판단 (보수적)")
    p.add_argument("--online", action="store_true",
                   help="모델이 없으면 HuggingFace 에서 직접 받는다 (기본: 오프라인, 캐시에 있는 모델만 사용)")
    return p.parse_args()


def setup_env(online: bool) -> None:
    """headroom 을 import 하기 전에 호출해야 한다."""
    if not online:
        os.environ.setdefault("HEADROOM_OFFLINE", "1")
    os.environ.setdefault("HEADROOM_TELEMETRY", "off")
    os.environ.setdefault("HEADROOM_BEACON", "off")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def print_model_help() -> None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        HF_HUB_CACHE = "~/.cache/huggingface/hub"
    print("\n[중단] Kompress 모델 파일이 이 PC에 없습니다.")
    print("       모델 없이 압축하면 Headroom 은 에러 없이 원문을 그대로 돌려주므로, 테스트를 멈춥니다.\n")
    print("인터넷이 되는 PC에서 아래 두 개를 받으세요 (버전 고정):")
    for repo, rev, desc in MODEL_REPOS:
        print(f"  # {desc}")
        print(f"  huggingface-cli download {repo} --revision {rev}")
    print("\n그다음 그 PC의 HuggingFace 캐시 폴더에서 아래 두 폴더를 이 PC의 같은 위치로 복사하세요:")
    for repo, _, _ in MODEL_REPOS:
        print(f"  models--{repo.replace('/', '--')}")
    print(f"\n이 PC의 캐시 위치: {HF_HUB_CACHE}")
    print("(회사 HuggingFace 미러가 있으면 HF_ENDPOINT=<미러주소> 로 지정하고 --online 으로 실행해도 됩니다.)")


def load_model(online: bool) -> str:
    """모델을 미리 불러온다. 성공하면 백엔드 이름(onnx/pytorch)을 돌려준다."""
    try:
        from headroom.transforms.kompress_compressor import KompressModelNotCached, _load_kompress
    except ImportError as e:
        print(f"[중단] headroom-ai 가 설치되어 있지 않습니다: {e}")
        print('       pip install "headroom-ai[proxy]"')
        sys.exit(2)

    try:
        _model, _tokenizer, backend = _load_kompress(allow_download=online)
    except KompressModelNotCached:
        print_model_help()
        sys.exit(2)
    except Exception as e:  # 패키지 누락, 손상된 모델 파일 등
        print(f"[중단] Kompress 모델을 불러오지 못했습니다: {type(e).__name__}: {e}")
        print('       pip install "headroom-ai[proxy]" 로 onnxruntime/transformers 가 설치되어 있는지 확인하세요.')
        sys.exit(2)
    return backend


def numbers_in(text: str) -> Counter:
    return Counter(_NUMBER_RE.findall(text))


def missing_numbers(original: str, compressed: str) -> list[tuple[str, int]]:
    """원문에 있던 숫자 중 압축본에서 (개수 기준으로) 사라진 것."""
    before, after = numbers_in(original), numbers_in(compressed)
    return sorted(((n, c - after.get(n, 0)) for n, c in before.items() if c > after.get(n, 0)),
                  key=lambda x: (-x[1], x[0]))


def removed_words(original: str, compressed: str, top: int = 30) -> list[tuple[str, int]]:
    """원문 대비 압축본에서 줄어든 단어 (많이 줄어든 순)."""
    before, after = Counter(original.split()), Counter(compressed.split())
    diff = Counter({w: c - after.get(w, 0) for w, c in before.items() if c > after.get(w, 0)})
    return diff.most_common(top)


def headings_lost(original: str, compressed: str) -> int:
    count = lambda t: sum(1 for line in t.splitlines() if line.lstrip().startswith("#"))
    return max(0, count(original) - count(compressed))


def write_report(path: Path, src: Path, result, elapsed: float, backend: str,
                 target_ratio: float | None) -> list[tuple[str, int]]:
    original, compressed = result.original, result.compressed
    lost_nums = missing_numbers(original, compressed)
    total_nums = sum(numbers_in(original).values())
    lines = [
        f"# Kompress 테스트 결과: {src.name}",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 백엔드 | {backend} |",
        f"| target_ratio | {target_ratio if target_ratio is not None else '자동 (Kompress 판단)'} |",
        f"| 토큰 (추정) | {result.original_tokens:,} → {result.compressed_tokens:,} |",
        f"| 절감률 | {result.savings_percentage:.1f}% |",
        f"| 글자 수 | {len(original):,} → {len(compressed):,} |",
        f"| 걸린 시간 | {elapsed:.2f}초 |",
        f"| 사라진 숫자 | {sum(c for _, c in lost_nums)} / {total_nums} 개 |",
        f"| 사라진 제목(#) 줄 | {headings_lost(original, compressed)} 개 |",
        "",
        "토큰 수는 headroom 의 추정값이다 (실제 Claude 토큰 수와 다를 수 있다).",
        "",
        "## 사라진 숫자",
        "",
    ]
    if lost_nums:
        lines += [f"- `{n}` × {c}" for n, c in lost_nums]
    else:
        lines.append("없음")
    lines += ["", "## 많이 지워진 단어 (상위 30개)", ""]
    words = removed_words(original, compressed)
    lines += [f"- `{w}` × {c}" for w, c in words] if words else ["없음"]
    lines += ["", "원문과 압축본을 나란히 열어, 숫자·날짜·조건·고유명사가 빠지지 않았는지 직접 확인할 것.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return lost_nums


def main() -> None:
    args = parse_args()
    setup_env(args.online)

    files = [f for f in args.files]
    for f in files:
        if not f.is_file():
            print(f"[중단] 파일이 없습니다: {f}")
            sys.exit(2)

    print("Kompress 모델 불러오는 중...")
    t0 = time.time()
    backend = load_model(args.online)
    print(f"  모델 준비 완료 ({backend}, {time.time() - t0:.1f}초)\n")

    from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig

    # CCR(복원 마커)은 끈다 — 이 앱은 압축본만 LLM 에 넘기고 복원 도구를 쓰지 않는다.
    compressor = KompressCompressor(KompressConfig(enable_ccr=False))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'파일':<30} {'토큰(추정)':>18} {'절감률':>7} {'사라진 숫자':>10} {'시간':>7}")
    for src in files:
        text = src.read_text(encoding="utf-8")
        t = time.time()
        result = compressor.compress(text, target_ratio=args.target_ratio, allow_download=args.online)
        elapsed = time.time() - t

        stem = src.stem
        (args.out_dir / f"{stem}.kompressed.md").write_text(result.compressed, encoding="utf-8")
        lost = write_report(args.out_dir / f"{stem}.report.md", src, result, elapsed, backend, args.target_ratio)

        tokens = f"{result.original_tokens:,} → {result.compressed_tokens:,}"
        print(f"{src.name:<30} {tokens:>18} {result.savings_percentage:>6.1f}% {sum(c for _, c in lost):>10} {elapsed:>6.2f}s")
        if result.compressed == text:
            print("  ! 압축본이 원문과 똑같습니다. 글이 너무 짧거나(64단어 미만), 압축 중 에러가 났을 수 있습니다. 위 로그를 확인하세요.")

    print(f"\n결과 저장: {args.out_dir}")


if __name__ == "__main__":
    main()
