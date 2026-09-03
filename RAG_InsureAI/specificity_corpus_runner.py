#!/usr/bin/env python3
"""
Specificity corpus runner — measures whether answers reshape available,
specific source detail (a named document list, a specific exclusions
list, etc.) into vague generic phrasing ("the required documents") that
drops it.

This is the regression harness for the Specificity Recall Guard (SRG),
built BEFORE the guard itself per the plan (PLAN_specificity_recall_
guard.md) — a prompt-only attempt at this exact problem was tried and
reverted the same session because it silently backfired (bloated answers
from 4-5 points to 7-8, added new filler lines, and the ONE targeted
phrase never actually changed) and was only caught by eyeballing a
single query. This corpus exists so "it looks right on the query I
tested" is never the bar again for this class of fix.

Modeled directly on contamination_corpus_runner.py's pattern: hand-
labeled ground truth (expected_items per enumerable case), NOT re-
derived from the same structural-detection logic the guard itself will
use — grading a filter with its own signal is circular. Reuses that
file's _ask()/_reset_state()/_assert_query_cache_disabled() conventions
verbatim (see that module's docstrings for why each exists) so the two
runners behave identically for anything they share.

Two categories of case:
  enumerable — query whose grounded answer SHOULD enumerate specific
               source items (documents, exclusions, covered events,
               claim steps). Metric: item recall — how many of the
               hand-verified expected_items actually appear in the
               answer (case-insensitive substring). This is the number
               the guard exists to raise.
  control    — definitional / single-fact / overview query with no
               strong source enumeration to lose. Tracked for point
               count and word count only. These must stay FLAT between
               a pre-guard and post-guard run — this is the bloat
               tripwire that would have caught the reverted prompt
               attempt immediately instead of after the fact.

Usage:
  python3 specificity_corpus_runner.py                       # full corpus
  python3 specificity_corpus_runner.py --category enumerable
  python3 specificity_corpus_runner.py --repeats 3
  python3 specificity_corpus_runner.py --out baseline.json
  python3 specificity_corpus_runner.py --compare baseline.json --out after.json
    (--compare prints the delta against a prior run's --out file — the
    actual before/after check this whole harness exists for)

Requires the api container running on :8501 with DISABLE_QUERY_CACHE=1
(same requirement, same reason, as contamination_corpus_runner.py — see
_assert_query_cache_disabled's own docstring).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

API_URL = "http://localhost:8501/ask-stream"
CONTAINER = "insurehub_api"
KV_CACHE_PATH = "/root/.insurehub/cache/query_kv_cache.json"

_HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_HERE, "specificity_corpus.json")

_CRASH_TEXT = "Could not generate an answer due to an internal error"


def _docker(cmd: str, capture: bool = False) -> str:
    try:
        res = subprocess.run(
            ["docker", "exec", CONTAINER, "sh", "-c", cmd],
            capture_output=True, text=True, timeout=30,
        )
        return res.stdout if capture else ""
    except Exception as exc:
        print(f"  [warn] docker exec failed: {exc}", file=sys.stderr)
        return ""


def _reset_state() -> None:
    _docker(f"rm -f {KV_CACHE_PATH}")


def _assert_query_cache_disabled() -> None:
    """See contamination_corpus_runner.py's identical function for the
    full rationale — deleting the cache FILE does not bust the in-memory
    copy a running process already holds; repeats need the env flag."""
    raw = _docker("printenv DISABLE_QUERY_CACHE 2>/dev/null", capture=True).strip()
    if raw.lower() not in ("1", "true", "yes"):
        print(
            f"ERROR: DISABLE_QUERY_CACHE={raw or '(unset)'} in container {CONTAINER!r} — "
            "the query KV cache is ENABLED.\n"
            "  Fix: set DISABLE_QUERY_CACHE=1 in RAG_InsureAI/.env, recreate the container\n"
            "  (docker compose up -d --force-recreate api), re-run, then restore afterwards.",
            file=sys.stderr,
        )
        sys.exit(2)


def _ask(query: str, session_id: str, timeout: int = 120) -> str:
    """POST to /ask-stream, return the FINAL corrected_text the user
    actually sees — see contamination_corpus_runner.py's _ask() for why
    this matters (the raw stream is pre-correction, corrected_text is a
    separate trailing field). Identical logic, duplicated rather than
    imported to keep this runner a standalone, dependency-free script
    matching the sibling runner's own convention."""
    body = json.dumps({"question": query, "session_id": session_id}).encode()
    req = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    idx = raw.rfind('\n\n{"sources"')
    if idx == -1:
        return raw
    pre_correction = raw[:idx]
    try:
        trailer = json.loads(raw[idx:].strip())
        corrected = trailer.get("corrected_text")
        if isinstance(corrected, str) and corrected.strip():
            return corrected
    except (json.JSONDecodeError, AttributeError):
        pass
    return pre_correction


_POINT_SPLIT_RE = re.compile(r'\n(?=\s*\d+\.\s)')
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])(?<!\d\.)\s+')


def _count_points(answer: str) -> int:
    """Numbered points if the answer uses them, else sentence count —
    same split convention the production pipeline's own filters use
    (see multi_source_rag.py's sentence-cap comment for the (?<!\\d\\.)
    numbered-marker exclusion reasoning)."""
    text = answer.strip()
    if not text:
        return 0
    if re.search(r'(?:^|\n)\s*\d+\.\s', text):
        return len([p for p in _POINT_SPLIT_RE.split(text) if p.strip()])
    return len([s for s in _SENT_SPLIT_RE.split(text) if s.strip()])


def _word_count(answer: str) -> int:
    return len(answer.split())


def _item_recall(answer: str, expected_items: list) -> tuple:
    """Case-insensitive substring match — deliberately simple and
    literal, mirroring the same "must be a literal substring" discipline
    the guard itself is required to use (see the plan's step 7). Returns
    (hits, total, matched_list)."""
    text = answer.lower()
    matched = [it for it in expected_items if it.lower() in text]
    return len(matched), len(expected_items), matched


def run_case(case: dict, repeats: int) -> dict:
    results = []
    for r in range(repeats):
        session = f"specificity-{case['id']}-{r}-{int(time.time())}"
        _reset_state()
        try:
            answer = _ask(case["query"], session)
            crashed = _CRASH_TEXT in answer
            entry = {
                "run": r,
                "crashed": crashed,
                "point_count": _count_points(answer),
                "word_count": _word_count(answer),
                "answer_preview": answer.strip()[:300],
            }
            if case["category"] == "enumerable":
                hits, total, matched = _item_recall(answer, case["expected_items"])
                entry["item_hits"] = hits
                entry["item_total"] = total
                entry["item_matched"] = matched
                entry["recall_pct"] = round(100.0 * hits / total, 1) if total else None
            results.append(entry)
        except Exception as exc:
            results.append({"run": r, "error": str(exc)})
        time.sleep(1)
    crashed_runs = sum(1 for x in results if x.get("crashed"))
    out = {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "repeats": repeats,
        "crashed_runs": crashed_runs,
        "any_crashed": crashed_runs > 0,
        "avg_point_count": round(sum(x.get("point_count", 0) for x in results) / len(results), 2) if results else 0,
        "avg_word_count": round(sum(x.get("word_count", 0) for x in results) / len(results), 1) if results else 0,
        "runs": results,
    }
    if case["category"] == "enumerable":
        recalls = [x["recall_pct"] for x in results if x.get("recall_pct") is not None]
        out["avg_recall_pct"] = round(sum(recalls) / len(recalls), 1) if recalls else None
        out["expected_items"] = case["expected_items"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["enumerable", "control"], help="filter to one category")
    ap.add_argument("--id", help="filter to cases whose id contains this substring")
    ap.add_argument("--repeats", type=int, default=1, help="samples per case (default 1)")
    ap.add_argument("--out", help="write full JSON results to this path")
    ap.add_argument("--compare", help="prior --out JSON file to diff this run against")
    args = ap.parse_args()

    _assert_query_cache_disabled()

    corpus = json.load(open(CORPUS_PATH))
    cases = corpus["cases"]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.id:
        cases = [c for c in cases if args.id in c["id"]]
    if not cases:
        print("No cases matched the filter.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(cases)} case(s) x {args.repeats} repeat(s) against {API_URL}\n")
    case_results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ({case['category']}) ...", flush=True)
        cr = run_case(case, args.repeats)
        case_results.append(cr)
        if cr.get("any_crashed"):
            print(f"    *** CRASHED ***")
        if cr["category"] == "enumerable":
            print(f"    recall={cr['avg_recall_pct']}%  points={cr['avg_point_count']}  words={cr['avg_word_count']}")
            for run in cr["runs"]:
                if run.get("item_matched") is not None:
                    missed = [it for it in cr["expected_items"] if it not in run["item_matched"]]
                    if missed:
                        print(f"      run{run['run']} missed: {missed}")
        else:
            print(f"    points={cr['avg_point_count']}  words={cr['avg_word_count']}")

    enumerable = [c for c in case_results if c["category"] == "enumerable"]
    controls = [c for c in case_results if c["category"] == "control"]

    def _avg(group, key):
        vals = [c[key] for c in group if c.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    crashed_cases = [c["id"] for c in case_results if c.get("any_crashed")]
    overall_recall = _avg(enumerable, "avg_recall_pct")
    enum_points = _avg(enumerable, "avg_point_count")
    enum_words = _avg(enumerable, "avg_word_count")
    ctrl_points = _avg(controls, "avg_point_count")
    ctrl_words = _avg(controls, "avg_word_count")

    print("\n" + "=" * 66)
    print("SPECIFICITY BASELINE")
    print("=" * 66)
    print(f"Crashes: {len(crashed_cases)} case(s)   <- MUST be 0   {crashed_cases if crashed_cases else ''}")
    print(f"Enumerable cases ({len(enumerable)}): avg item recall = {overall_recall}%")
    print(f"  avg points/answer = {enum_points}   avg words/answer = {enum_words}")
    print(f"Control cases ({len(controls)}): avg points/answer = {ctrl_points}   avg words/answer = {ctrl_words}")
    for c in enumerable:
        print(f"  {c['id']}: recall={c['avg_recall_pct']}%  points={c['avg_point_count']}  words={c['avg_word_count']}")
    print("=" * 66)

    payload = {
        "timestamp": time.time(),
        "repeats": args.repeats,
        "crashed_case_ids": crashed_cases,
        "overall_recall_pct": overall_recall,
        "enumerable_avg_points": enum_points,
        "enumerable_avg_words": enum_words,
        "control_avg_points": ctrl_points,
        "control_avg_words": ctrl_words,
        "cases": case_results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results -> {args.out}")

    if args.compare:
        try:
            prior = json.load(open(args.compare))
        except Exception as exc:
            print(f"\n[compare] could not load {args.compare}: {exc}", file=sys.stderr)
            sys.exit(1)
        print("\n" + "=" * 66)
        print(f"COMPARE vs {args.compare}")
        print("=" * 66)
        d_recall = (overall_recall or 0) - (prior.get("overall_recall_pct") or 0)
        d_enum_pts = (enum_points or 0) - (prior.get("enumerable_avg_points") or 0)
        d_enum_words = (enum_words or 0) - (prior.get("enumerable_avg_words") or 0)
        d_ctrl_pts = (ctrl_points or 0) - (prior.get("control_avg_points") or 0)
        d_ctrl_words = (ctrl_words or 0) - (prior.get("control_avg_words") or 0)
        print(f"Item recall:            {prior.get('overall_recall_pct')}% -> {overall_recall}%  (delta {d_recall:+.1f})   <- should INCREASE")
        print(f"Enumerable avg points:  {prior.get('enumerable_avg_points')} -> {enum_points}  (delta {d_enum_pts:+.2f})   <- should stay ~flat")
        print(f"Enumerable avg words:   {prior.get('enumerable_avg_words')} -> {enum_words}  (delta {d_enum_words:+.1f})   <- should NOT jump")
        print(f"Control avg points:     {prior.get('control_avg_points')} -> {ctrl_points}  (delta {d_ctrl_pts:+.2f})   <- MUST stay ~flat")
        print(f"Control avg words:      {prior.get('control_avg_words')} -> {ctrl_words}  (delta {d_ctrl_words:+.1f})   <- MUST stay ~flat")
        print()
        _hard_fail = abs(d_ctrl_pts) > 0.5 or abs(d_ctrl_words) > 15
        _bloat_warn = d_enum_pts > 1.0 or d_enum_words > 25
        _no_improvement = d_recall <= 0
        if crashed_cases:
            print("*** FAIL — crash(es) in this run ***")
        elif _hard_fail:
            print("*** FAIL — control cases drifted (points or words moved materially on cases with nothing to fix) ***")
        elif _bloat_warn:
            print("*** FAIL — enumerable answers got materially longer, not just more complete (bloat regression, same shape as the reverted prompt attempt) ***")
        elif _no_improvement:
            print("*** FAIL — item recall did not improve ***")
        else:
            print(f"PASS — recall improved by {d_recall:+.1f}pp with no material bloat and controls held flat")
        sys.exit(1 if (crashed_cases or _hard_fail or _bloat_warn or _no_improvement) else 0)


if __name__ == "__main__":
    main()
