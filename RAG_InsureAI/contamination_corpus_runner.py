#!/usr/bin/env python3
"""
Contamination corpus runner — Phase 0c of the cross-topic-contamination
root-cause plan (plan.md at the repo root).

Runs contamination_corpus.json against the live backend and emits ONE
north-star number: the contamination rate (% of answers containing at
least one forbidden, off-topic phrase). This is the metric every later
phase has to move — a fix is never "done" because one query looks good,
only because this rate drops on the whole corpus without the clean/
exemption controls regressing.

It also collects the per-point cross-encoder relevance scores from the
contamination trace (contamination_trace.py), which is the raw material
for calibrating the Phase-2 topic-relevance threshold — separating the
score band of genuinely off-topic points from on-topic ones.

Design notes:
  * Before every request it purges the query KV cache and truncates the
    trace file inside the container, so (a) repeat queries in the corpus
    are real fresh samples of the model's nondeterminism rather than
    cache hits, and (b) each request's trace record can be read back
    unambiguously. This is the established "purge query_kv_cache.json
    before retesting" gotcha, applied per-request.
  * Contamination is scored by labeled forbidden_phrases (ground truth
    from real observed failures), NOT by the semantic signal — the
    semantic signal is what we're validating, so it can't also be the
    grader. must_allow_phrases / allowed_despite_topic_match are honored
    so a legitimately on-topic mention of a term is never counted.
  * Requires the api container running on :8501 with CONTAMINATION_TRACE=1.

Usage:
  python3 contamination_corpus_runner.py                  # full corpus
  python3 contamination_corpus_runner.py --category known_contamination_repro
  python3 contamination_corpus_runner.py --id motor       # id substring
  python3 contamination_corpus_runner.py --repeats 3      # N samples/case
  python3 contamination_corpus_runner.py --out baseline.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

API_URL = "http://localhost:8501/ask-stream"
CONTAINER = "insurehub_api"
KV_CACHE_PATH = "/root/.insurehub/cache/query_kv_cache.json"
TRACE_PATH = "/root/.insurehub/contamination_trace/trace.jsonl"

_HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_HERE, "contamination_corpus.json")


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
    _docker(f"rm -f {KV_CACHE_PATH} {TRACE_PATH}")


def _read_trace_record() -> dict:
    out = _docker(f"cat {TRACE_PATH} 2>/dev/null", capture=True).strip()
    if not out:
        return {}
    # One request -> at most one trace line; take the last if more.
    line = out.splitlines()[-1]
    try:
        return json.loads(line)
    except Exception:
        return {}


def _ask(query: str, session_id: str, timeout: int = 120) -> str:
    """POST to /ask-stream, return the answer text (the streamed body
    with the trailing '\\n\\n{"sources": ...}' metadata blob stripped).
    """
    body = json.dumps({"question": query, "session_id": session_id}).encode()
    req = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    idx = raw.rfind('\n\n{"sources"')
    return raw[:idx] if idx != -1 else raw


def _turns(query: str) -> list:
    """A corpus query may encode a multi-turn conversation as
    'first turn -> second turn' (e.g. the third-party-victim example
    case). Split into ordered turns; the contamination check applies to
    the LAST turn's answer.
    """
    if "->" in query:
        return [t.strip() for t in query.split("->") if t.strip()]
    return [query]


def _find_forbidden(answer: str, case: dict) -> list:
    """Return the list of forbidden phrases actually present in *answer*,
    after removing any that the case explicitly whitelists (a phrase that
    is legitimately on-topic here despite matching a forbidden pattern).
    """
    text = answer.lower()
    allow = set()
    for k in ("must_allow_phrases", "allowed_despite_topic_match"):
        for p in case.get(k, []) or []:
            allow.add(p.lower())
    hits = []
    for phrase in case.get("forbidden_phrases", []) or []:
        p = phrase.lower()
        if p in allow:
            continue
        if p in text:
            hits.append(phrase)
    return hits


def run_case(case: dict, repeats: int) -> dict:
    results = []
    for r in range(repeats):
        session = f"corpus-{case['id']}-{r}-{int(time.time())}"
        _reset_state()
        try:
            answer = ""
            for turn in _turns(case["query"]):
                answer = _ask(turn, session)
            trace = _read_trace_record()
            hits = _find_forbidden(answer, case)
            point_scores = [
                p.get("relevance_score")
                for p in trace.get("points", [])
                if isinstance(p.get("relevance_score"), (int, float))
            ]
            # original_point_count is what the model actually generated;
            # final_point_count is what survived (contamination_trace.py
            # writes both). Their difference is what Phase 2's plan calls
            # "legitimate points dropped" when this case is a clean/
            # exemption control — the contamination boolean above only
            # catches forbidden-phrase leaks, not a gate over-firing on a
            # correct answer, which is exactly the failure mode the OLD
            # ratio-to-max gate design had (see project memory).
            orig_n = trace.get("original_point_count")
            final_n = trace.get("final_point_count", len(trace.get("points", [])))
            points_dropped = (orig_n - final_n) if isinstance(orig_n, int) else None
            # The guarded gate (2026-07-23) demotes instead of deleting, so
            # original_point_count == final_point_count even when it fires —
            # points_dropped above is now expected to always be 0 for this
            # specific mechanism. point_relevance_demoted is the real signal:
            # which points (if any) this gate moved, and what type it
            # confirmed them as. A demotion on a clean/exemption control is
            # the failure mode to watch for, same spirit as points_dropped
            # was for the deleting version.
            demoted = trace.get("point_relevance_demoted") or []
            results.append({
                "run": r,
                "contaminated": bool(hits),
                "forbidden_hits": hits,
                "point_count": len(trace.get("points", [])),
                "original_point_count": orig_n,
                "final_point_count": final_n,
                "points_dropped": points_dropped,
                "point_relevance_demoted": demoted,
                "point_scores": point_scores,
                "retrieved_types": [c.get("policy_type") for c in trace.get("retrieved_chunks", [])],
                "answer_preview": answer.strip()[:280],
            })
        except Exception as exc:
            results.append({"run": r, "error": str(exc)})
        # Small gap so we never hammer the single-worker backend.
        time.sleep(1)
    contaminated_runs = sum(1 for x in results if x.get("contaminated"))
    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_topic": case.get("expected_topic"),
        "repeats": repeats,
        "contaminated_runs": contaminated_runs,
        "any_contaminated": contaminated_runs > 0,
        "runs": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", help="filter to one category")
    ap.add_argument("--id", help="filter to cases whose id contains this substring")
    ap.add_argument("--repeats", type=int, default=1, help="samples per case (default 1)")
    ap.add_argument("--out", help="write full JSON results to this path")
    args = ap.parse_args()

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
        if cr["any_contaminated"]:
            for run in cr["runs"]:
                if run.get("contaminated"):
                    print(f"    CONTAMINATED (run {run['run']}): {run['forbidden_hits']}")
                    print(f"      -> {run['answer_preview'][:160]}")
        else:
            print("    clean")

    # ── Aggregate ──────────────────────────────────────────────────────
    repro = [c for c in case_results if c["category"] == "known_contamination_repro"]
    controls = [c for c in case_results if c["category"] in ("clean_control", "exemption_control")]
    narrative = [c for c in case_results if c["category"] == "narrative_structure_repro"]

    def _rate(group):
        if not group:
            return 0.0, 0, 0
        total_runs = sum(c["repeats"] for c in group)
        contaminated_runs = sum(c["contaminated_runs"] for c in group)
        return (100.0 * contaminated_runs / total_runs if total_runs else 0.0), contaminated_runs, total_runs

    all_rate, all_c, all_t = _rate(case_results)
    repro_rate, repro_c, repro_t = _rate(repro)
    ctrl_rate, ctrl_c, ctrl_t = _rate(controls)

    # Points-dropped audit. IMPORTANT — as of the 2026-07-23 guarded gate,
    # this number is NOT attributable to the Phase-2 point-relevance gate;
    # that gate demotes (reorders), it structurally cannot reduce
    # original_point_count -> final_point_count by construction. Verified
    # live: reproducing one of this sweep's drop cases showed
    # point_relevance_demoted=[] in the trace while the log showed
    # "dropped 1 cross-topic-contaminated point(s)" and "dropped sentence/
    # point claiming fines coverage" firing instead — pre-existing filters
    # (_text_has_giveaway_contamination and the fines-claim check) that
    # exist independently of this plan's Phase 2 work and were already
    # active in the gate-OFF baseline (which happened to show 0 drops
    # simply because that batch's specific generations didn't trigger
    # them). Kept here as a general "did total point count change" signal
    # — useful, but read it as "some correction fired," not "the gate did
    # this." point_relevance_demoted below is the gate-specific signal.
    def _drop_events(group):
        events = []
        for c in group:
            for run in c["runs"]:
                d = run.get("points_dropped")
                if isinstance(d, int) and d > 0:
                    events.append({"id": c["id"], "run": run["run"], "dropped": d,
                                    "original": run.get("original_point_count"),
                                    "final": run.get("final_point_count")})
        return events

    control_drops = _drop_events(controls)
    repro_drops = _drop_events(repro)

    # Demote audit — the 2026-07-23 guarded gate reorders instead of
    # deleting, so points_dropped above will be 0 for it even when it
    # fires. This is the equivalent check for the new mechanism: any
    # demotion on a clean/exemption control is a false positive (the
    # guards should have prevented it), any demotion on a repro case is
    # the gate doing its job.
    def _demote_events(group):
        events = []
        for c in group:
            for run in c["runs"]:
                for d in run.get("point_relevance_demoted") or []:
                    events.append({"id": c["id"], "run": run["run"],
                                    "confirmed_type": d.get("confirmed_type"),
                                    "text": d.get("text")})
        return events

    control_demotes = _demote_events(controls)
    repro_demotes = _demote_events(repro)

    # Score-band split for Phase-2 calibration: relevance scores of points
    # in answers that DID contaminate vs. those that stayed clean.
    contam_scores, clean_scores = [], []
    for c in case_results:
        for run in c["runs"]:
            for s in run.get("point_scores", []):
                (contam_scores if run.get("contaminated") else clean_scores).append(s)

    def _pctiles(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        n = len(xs)
        return {
            "n": n,
            "min": round(xs[0], 4),
            "p10": round(xs[max(0, n // 10)], 4),
            "median": round(xs[n // 2], 4),
            "p90": round(xs[min(n - 1, 9 * n // 10)], 4),
            "max": round(xs[-1], 4),
        }

    print("\n" + "=" * 66)
    print("CONTAMINATION BASELINE")
    print("=" * 66)
    print(f"Overall contamination rate:        {all_rate:.1f}%  ({all_c}/{all_t} runs)")
    print(f"  known_contamination_repro:       {repro_rate:.1f}%  ({repro_c}/{repro_t} runs)")
    print(f"  clean+exemption controls:        {ctrl_rate:.1f}%  ({ctrl_c}/{ctrl_t} runs)   <- MUST stay ~0")
    if narrative:
        n_rate, n_c, n_t = _rate(narrative)
        print(f"  narrative_structure_repro:       {n_rate:.1f}%  ({n_c}/{n_t} runs)")
    print()
    print(f"Total point-count reduction on clean+exemption controls: {len(control_drops)}  "
          f"(NOT necessarily the Phase-2 gate — see point_relevance_demoted below for that)")
    if control_drops:
        print("  Some correction mechanism removed a point on a control case. Check")
        print("  point_relevance_demoted below to see if it was THIS gate or another filter.")
        for e in control_drops:
            print(f"    {e['id']} run{e['run']}: dropped {e['dropped']} of {e['original']} -> {e['final']} left")
    if repro_drops:
        print(f"Points dropped on known_contamination_repro (expected/good): {len(repro_drops)}")
        for e in repro_drops:
            print(f"    {e['id']} run{e['run']}: dropped {e['dropped']} of {e['original']} -> {e['final']} left")
    print()
    print(f"Points DEMOTED on clean+exemption controls: {len(control_demotes)}   <- MUST be 0")
    if control_demotes:
        print("  *** GATE DEMOTED A POINT ON A CONTROL CASE — the guards failed to prevent this ***")
        for e in control_demotes:
            print(f"    {e['id']} run{e['run']}: confirmed_type={e['confirmed_type']!r} text={e['text'][:90]!r}")
    if repro_demotes:
        print(f"Points demoted on known_contamination_repro (expected/good): {len(repro_demotes)}")
        for e in repro_demotes:
            print(f"    {e['id']} run{e['run']}: confirmed_type={e['confirmed_type']!r} text={e['text'][:90]!r}")
    print()
    print("Per-point relevance score bands (for Phase-2 threshold calibration):")
    print(f"  points in CONTAMINATED answers:  {_pctiles(contam_scores)}")
    print(f"  points in CLEAN answers:         {_pctiles(clean_scores)}")
    print("=" * 66)

    payload = {
        "timestamp": time.time(),
        "api_url": API_URL,
        "case_count": len(case_results),
        "repeats": args.repeats,
        "overall_rate_pct": round(all_rate, 2),
        "repro_rate_pct": round(repro_rate, 2),
        "control_rate_pct": round(ctrl_rate, 2),
        "control_point_drops": control_drops,
        "repro_point_drops": repro_drops,
        "control_point_demotes": control_demotes,
        "repro_point_demotes": repro_demotes,
        "contaminated_point_scores": _pctiles(contam_scores),
        "clean_point_scores": _pctiles(clean_scores),
        "cases": case_results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results -> {args.out}")


if __name__ == "__main__":
    main()
