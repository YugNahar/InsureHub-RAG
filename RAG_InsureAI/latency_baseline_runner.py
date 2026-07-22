#!/usr/bin/env python3
"""
Latency baseline runner — Phase 0/4 of the RAG latency plan
(plan_latency.md at the repo root).

Runs a small FIXED query set (2 brief, 2 detailed, 1 follow-up, 1
refusal) against the live backend and prints a clean per-phase +
TTFT table. This is the "measure clean before touching anything" step
every later phase (GPU device loading, compressor changes, output-
length shaping) must be compared against — and the repeatable command
Phase 4 asks for, so it can be re-run on the GPU server once deployed
there without rewriting anything.

Design notes:
  * Each query is sent TWICE and only the SECOND run is reported as the
    baseline data point ("warm") — the first run is discarded but still
    printed for reference. This warms the context-compressor's per-chunk
    sentence-embedding cache (context_compressor.py's _sent_cache, keyed
    by chunk TEXT not by query) without needing DISABLE_QUERY_CACHE=1 to
    ALSO warm the model-weight/OS caches. Requires DISABLE_QUERY_CACHE=1
    in the container's env, or the second run just hits the exact-match
    KV cache and returns near-instantly — that's a different, degenerate
    code path, not what "warm" means here. This script checks for that
    env var and warns loudly if it isn't set.
  * TIMING lines (including ttft=) are read back from `docker logs`
    right after each request completes. Matched on the log line's
    question=%r field, NOT query=%r (which is retrieval_query — rewritten
    for follow-ups, typo correction, and query cleaning; matching on it
    silently produced zero matches for the follow-up case). question= is
    the raw, rarely-reassigned input text and was added to the TIMING
    line specifically so this harness has a stable match target.
  * A FRESH session_id is used for every (case, pass) — cold and warm
    are two independent conversations, not two turns of the same one, so
    a multi-turn case's warm pass starts from a clean slate exactly like
    its cold pass did, not with the cold pass's history still attached.
  * Follow-up queries need real prior turns to be meaningful — the
    follow-up case sends a context-setting first message before the
    actual follow-up (the measured one, per-pass, in the same session).
  * A run with ZERO TIMING-line matches is treated as a hard failure
    (nonzero exit, loud error), not silently printed as "?" — this is
    exactly what would have happened if this script were run before the
    code emitting ttft=/question= was ever deployed to the container.

Usage:
  python3 latency_baseline_runner.py                 # full 6-query set
  python3 latency_baseline_runner.py --out baseline.json
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid

API_URL = "http://localhost:8501/ask-stream"
CONTAINER = "insurehub_api"
KV_CACHE_PATH = "/root/.insurehub/cache/query_kv_cache.json"

# (label, mode, turns) — turns is a list of message strings; only the
# LAST turn's TIMING line is the measured data point for multi-turn cases.
QUERY_SET = [
    ("brief_1", "brief", ["What is personal accident insurance?"]),
    ("brief_2", "brief", ["What does a comprehensive motor policy cover?"]),
    ("detailed_1", "detailed", ["Explain fire insurance in detail"]),
    ("detailed_2", "detailed", ["Explain motor insurance in detail"]),
    ("followup", "followup", [
        "What is personal accident insurance?",
        "What's excluded under it?",
    ]),
    ("refusal", "refusal", ["What is the exact premium for a 1998 Yugo GV in Alaska?"]),
]


def _docker(cmd: str) -> str:
    try:
        res = subprocess.run(
            ["docker", "exec", CONTAINER, "sh", "-c", cmd],
            capture_output=True, text=True, timeout=30,
        )
        return res.stdout
    except Exception as exc:
        print(f"  [warn] docker exec failed: {exc}", file=sys.stderr)
        return ""


def _check_disable_query_cache() -> None:
    out = _docker("echo $DISABLE_QUERY_CACHE").strip()
    if out != "1":
        print(
            "WARNING: DISABLE_QUERY_CACHE is not set to 1 in the container "
            "(got %r) — the 'warm' second run of each query will likely hit "
            "the exact-match KV cache and return near-instantly instead of "
            "genuinely re-running retrieval/reranking/generation. Results "
            "below are NOT a valid baseline until this is set." % out,
            file=sys.stderr,
        )


def _purge_kv_cache() -> None:
    _docker(f"rm -f {KV_CACHE_PATH}")


def _ask(query: str, session_id: str, timeout: int = 120) -> str:
    body = json.dumps({"question": query, "session_id": session_id}).encode()
    req = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


_TIMING_RE = re.compile(
    r"TIMING total=(\d+)ms ttft=(\S+) retrieval=(\S+) grounding=(\S+) llm=(\S+) other=(\d+)ms "
    r"preprocess=(\S+) promptbuild=(\S+) postllm=(\S+) detailed=(\S+) query=(?:'[^']*'|\"[^\"]*\") "
    r"question=(.+)$"
)


def _ms(s: str):
    if s in ("n/a", None):
        return None
    return int(s.rstrip("ms"))


class TimingNotFound(RuntimeError):
    pass


def _read_timing_for_question(question: str, tail_lines: int = 300) -> dict:
    """Grep the container's recent logs for the TIMING line whose
    question=%r field matches *question* (the ORIGINAL text this script
    sent — see module docstring for why this, not query=, is the match
    target). Takes the LAST match in the window — the most recent request
    for that exact text. Raises TimingNotFound on zero matches rather
    than returning {} silently (Opus review finding B2: silent zero-match
    printed a table of '?' with no indication the run was invalid)."""
    # This app's Python logging (including every TIMING line) goes to
    # stderr, not stdout — `capture_output=True` splits the two, so
    # reading only `.stdout` silently misses every match. Merge them the
    # same way `docker logs ... 2>&1` on a terminal would.
    logs = subprocess.run(
        ["docker", "logs", "--tail", str(tail_lines), CONTAINER],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15,
    ).stdout
    needle = question[:80]
    matches = []
    for line in logs.splitlines():
        if "TIMING" not in line or needle not in line:
            continue
        m = _TIMING_RE.search(line)
        if m:
            matches.append(m)
    if not matches:
        raise TimingNotFound(
            f"No TIMING line found matching question={question[:80]!r} in the last "
            f"{tail_lines} log lines. Either the code emitting ttft=/question= isn't "
            f"actually running in the container (restart it and verify with a single "
            f"manual request first), or this request hit an early-return path that "
            f"doesn't log TIMING at all."
        )
    m = matches[-1]
    return {
        "total_ms": _ms(m.group(1)),
        "ttft_ms": _ms(m.group(2)),
        "retrieval_ms": _ms(m.group(3)),
        "grounding_ms": _ms(m.group(4)),
        "llm_ms": _ms(m.group(5)),
        "other_ms": _ms(m.group(6)),
        "preprocess_ms": _ms(m.group(7)),
        "promptbuild_ms": _ms(m.group(8)),
        "postllm_ms": _ms(m.group(9)),
        "detailed": m.group(10),
    }


def run_case(label: str, mode: str, turns: list) -> dict:
    print(f"[{label}] ({mode}) — {turns[-1]!r}")

    runs = []
    ok = True
    for pass_name in ("cold", "warm"):
        # Fresh session per PASS, not shared across cold/warm — a multi-
        # turn case's warm pass must start from a clean slate exactly like
        # its cold pass did (Opus review D3: sharing one session_id meant
        # the warm pass ran as turns 3-4 with the cold pass's history
        # still attached, measuring a different conversational state).
        session_id = f"latency-baseline-{label}-{pass_name}-{uuid.uuid4().hex[:8]}"
        _purge_kv_cache()
        t0 = time.time()
        try:
            for turn in turns:
                _ask(turn, session_id)
        except Exception as exc:
            print(f"    [{pass_name}] request failed: {exc}", file=sys.stderr)
            ok = False
            continue
        wall_ms = round((time.time() - t0) * 1000)
        try:
            timing = _read_timing_for_question(turns[-1])
        except TimingNotFound as exc:
            print(f"    [{pass_name}] FAILED: {exc}", file=sys.stderr)
            ok = False
            timing = {}
        timing["wall_ms"] = wall_ms
        runs.append((pass_name, timing))
        if timing:
            print(
                f"    [{pass_name}] total={timing.get('total_ms')}ms "
                f"ttft={timing.get('ttft_ms')}ms llm={timing.get('llm_ms')}ms "
                f"retrieval={timing.get('retrieval_ms')}ms"
            )

    warm = next((t for name, t in runs if name == "warm"), {})
    cold = next((t for name, t in runs if name == "cold"), {})
    return {"label": label, "mode": mode, "query": turns[-1], "cold": cold, "warm": warm, "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args()

    _check_disable_query_cache()

    results = [run_case(label, mode, turns) for label, mode, turns in QUERY_SET]

    failed = [r["label"] for r in results if not r["ok"]]

    print("\n" + "=" * 78)
    print("LATENCY BASELINE (warm run — this is the number that matters)")
    print("=" * 78)
    header = f"{'label':<12} {'mode':<10} {'total':>7} {'ttft':>7} {'retr':>7} {'ground':>7} {'llm':>7} {'promptbuild':>12}"
    print(header)

    def _cell(v):
        # A field can be legitimately None (e.g. the refusal case never
        # runs llm/promptbuild — not a failure, just a phase that didn't
        # execute) — Opus review finding M1: printing raw None instead of
        # a display placeholder.
        return "n/a" if v is None else str(v)

    for r in results:
        w = r["warm"]
        # Check the explicit ok flag, not `if not w:` — timing["wall_ms"]
        # is set unconditionally after every pass (see run_case), so a
        # failed pass's dict is never actually empty and that check never
        # caught the failure, crashing the format string below on None.
        if not r["ok"] or w.get("total_ms") is None:
            print(f"{r['label']:<12} {r['mode']:<10} {'FAILED — see stderr above':>60}")
            continue
        print(
            f"{r['label']:<12} {r['mode']:<10} "
            f"{_cell(w.get('total_ms')):>7} {_cell(w.get('ttft_ms')):>7} "
            f"{_cell(w.get('retrieval_ms')):>7} {_cell(w.get('grounding_ms')):>7} "
            f"{_cell(w.get('llm_ms')):>7} {_cell(w.get('promptbuild_ms')):>12}"
        )
    print("=" * 78)

    if failed:
        print(
            f"\n{len(failed)}/{len(results)} case(s) FAILED to produce a TIMING match: "
            f"{failed} — this is NOT a valid baseline. See stderr above for why.",
            file=sys.stderr,
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"timestamp": time.time(), "cases": results, "failed": failed}, f, indent=2)
        print(f"\nFull results -> {args.out}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
