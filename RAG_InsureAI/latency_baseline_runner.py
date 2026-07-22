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
  * Each query gets ONE discarded warm-up request followed by REPEATS
    measured ones, and the reported number is the MEDIAN of the measured
    set (min-max spread is printed alongside and stored in the JSON).
    The warm-up is what the plan's "run twice, take the 2nd" asks for:
    it fills the context-compressor's per-chunk sentence-embedding cache
    (context_compressor.py's _sent_cache, keyed by chunk TEXT not by
    query) for this query's chunks. Requires DISABLE_QUERY_CACHE=1 in
    the container's env, or every repeat after the first just hits the
    exact-match KV cache and returns near-instantly — a different,
    degenerate code path. This script checks and warns loudly if unset.
  * Repeats exist because a single sample is not a measurement here.
    The generation backend is a REMOTE, shared vLLM host (VLLM_HOST),
    so "idle backend" cannot be enforced from this side. Two runs of
    this script with zero code change between them produced a 3.6x
    swing on one case (brief_1 total 55.3s then 15.2s). Phase 4 wants
    to attribute a before/after delta to a code change; with n=1 an
    outlier is indistinguishable from a real regression.
  * There is deliberately NO cold/warm split. It was tried and measured
    nothing: _purge_kv_cache() clears query_kv_cache.json only, while
    the compressor's in-process _sent_cache — the thing "warm" was
    supposed to warm — is never reset between passes, cases, or runs
    (logs show it accumulating monotonically and surviving across
    runs), so the "cold" pass was already warm. Retrieval time moved
    -8/-15/+8/-4/-6/+1 percent cold->warm across the six cases: noise.
    The doubled runtime now buys repeats instead.
  * Answer length is recorded per run. Phase 3 of the plan is output-
    length/prompt shaping, whose exit criterion is "tighter/complete,
    total unchanged-or-better" — unmeasurable without it. It also
    de-confounds llm=: at the remote host's ~7-8 tok/s a 512-token cap
    is ~64s, so "the system got slower" and "the model wrote more" look
    identical in the timing alone.
  * TIMING lines (including ttft=) are read back from `docker logs`
    right after each request completes. Matched on the log line's
    question=%r field, NOT query=%r (which is retrieval_query — rewritten
    for follow-ups, typo correction, and query cleaning; matching on it
    silently produced zero matches for the follow-up case). question= is
    the raw, rarely-reassigned input text and was added to the TIMING
    line specifically so this harness has a stable match target.
  * A FRESH session_id is used for every single pass (warm-up and each
    repeat) — every pass is an independent conversation, so a multi-turn
    case's later repeats start from a clean slate exactly like its first
    did rather than accumulating the previous passes' history.
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
        # Confirm the text matched in the CAPTURED question= field, not
        # just somewhere in the line — query= carries text too, so a raw
        # substring test can match a different request whose retrieval_
        # query happens to contain this question. The set already holds
        # duplicate text (brief_1 is byte-identical to followup's first
        # turn), so this is one reordering away from mattering.
        if m and needle in m.group(11):
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


_PHASE_KEYS = (
    "total_ms", "ttft_ms", "retrieval_ms", "grounding_ms", "llm_ms",
    "other_ms", "preprocess_ms", "promptbuild_ms", "postllm_ms",
    "answer_chars",
)


def _median_of(runs: list, key: str):
    """Median of a phase across the measured repeats, ignoring runs where
    the phase legitimately didn't execute (llm/promptbuild on a refusal)."""
    vals = sorted(r[key] for r in runs if r.get(key) is not None)
    if not vals:
        return None
    return vals[len(vals) // 2] if len(vals) % 2 else round((vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2)


def _one_pass(label: str, turns: list, tag: str) -> dict:
    """One full request sequence + its TIMING readback. Fresh session so
    no pass inherits a previous pass's conversation history."""
    session_id = f"latency-baseline-{label}-{tag}-{uuid.uuid4().hex[:8]}"
    _purge_kv_cache()
    t0 = time.time()
    answer = ""
    for turn in turns:
        answer = _ask(turn, session_id)
    wall_ms = round((time.time() - t0) * 1000)
    timing = _read_timing_for_question(turns[-1])
    timing["wall_ms"] = wall_ms
    # Output length, so Phase 3 (output-length shaping) is measurable and
    # llm= isn't confounded by how much the model happened to write.
    # Strips the trailing JSON sources/done payload the stream appends.
    timing["answer_chars"] = len(answer.split('\n\n{"sources"')[0].strip())
    return timing


def run_case(label: str, mode: str, turns: list, repeats: int) -> dict:
    print(f"[{label}] ({mode}) — {turns[-1]!r}")

    ok = True
    runs = []
    try:
        # Discarded warm-up: fills the compressor's per-chunk sentence
        # cache for THIS query's chunks so the measured repeats aren't
        # measuring first-touch embedding cost.
        _one_pass(label, turns, "warmup")
        for i in range(repeats):
            t = _one_pass(label, turns, f"rep{i + 1}")
            runs.append(t)
            print(
                f"    [rep{i + 1}] total={t.get('total_ms')}ms ttft={t.get('ttft_ms')}ms "
                f"llm={t.get('llm_ms')}ms retrieval={t.get('retrieval_ms')}ms "
                f"chars={t.get('answer_chars')}"
            )
    except TimingNotFound as exc:
        print(f"    FAILED: {exc}", file=sys.stderr)
        ok = False
    except Exception as exc:
        print(f"    request failed: {exc}", file=sys.stderr)
        ok = False

    median = {k: _median_of(runs, k) for k in _PHASE_KEYS} if runs else {}
    totals = [r["total_ms"] for r in runs if r.get("total_ms") is not None]
    if totals:
        median["total_min_ms"], median["total_max_ms"] = min(totals), max(totals)
        # Spread is reported next to every median so a number that happens
        # to sit on a noisy case can't be mistaken for a precise one.
        median["spread_x"] = round(max(totals) / max(min(totals), 1), 2)
    return {
        "label": label, "mode": mode, "query": turns[-1],
        "repeats": len(runs), "median": median, "runs": runs, "ok": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write full results JSON here")
    ap.add_argument("--repeats", type=int, default=3,
                    help="measured repeats per query after one discarded warm-up (default 3)")
    args = ap.parse_args()

    _check_disable_query_cache()

    results = [run_case(label, mode, turns, args.repeats)
               for label, mode, turns in QUERY_SET]

    failed = [r["label"] for r in results if not r["ok"]]

    print("\n" + "=" * 96)
    print(f"LATENCY BASELINE — median of {args.repeats} measured repeats (ms)")
    print("=" * 96)
    header = (f"{'label':<12} {'mode':<10} {'total':>7} {'ttft':>7} {'retr':>7} "
              f"{'ground':>7} {'llm':>7} {'other':>7} {'chars':>6} {'spread':>7}")
    print(header)

    def _cell(v):
        # A field can be legitimately None (e.g. the refusal case never
        # runs llm/promptbuild — not a failure, just a phase that didn't
        # execute) — Opus review finding M1: printing raw None instead of
        # a display placeholder.
        return "n/a" if v is None else str(v)

    for r in results:
        w = r["median"]
        if not r["ok"] or w.get("total_ms") is None:
            print(f"{r['label']:<12} {r['mode']:<10} {'FAILED — see stderr above':>60}")
            continue
        print(
            f"{r['label']:<12} {r['mode']:<10} "
            f"{_cell(w.get('total_ms')):>7} {_cell(w.get('ttft_ms')):>7} "
            f"{_cell(w.get('retrieval_ms')):>7} {_cell(w.get('grounding_ms')):>7} "
            f"{_cell(w.get('llm_ms')):>7} {_cell(w.get('other_ms')):>7} "
            f"{_cell(w.get('answer_chars')):>6} {str(w.get('spread_x', '?')) + 'x':>7}"
        )
    print("=" * 96)
    print("ttft is a PREFIX of total on generation rows, but EQUALS total on the")
    print("refusal row (nothing streams there) — do not average the column.")
    noisy = [r["label"] for r in results if (r["median"].get("spread_x") or 0) >= 1.5]
    if noisy:
        print(f"\nWARNING: {noisy} varied >=1.5x across repeats — treat those medians as soft.")

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
