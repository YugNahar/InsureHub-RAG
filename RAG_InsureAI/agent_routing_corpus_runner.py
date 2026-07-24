#!/usr/bin/env python3
"""
Agent-routing regression corpus — the "clean control set" for the
agent_router package (RAG_InsureAI/app/agent_router/), mirroring the
established pattern from contamination_corpus_runner.py: purge-before-run,
per-case JSON corpus, PASS/FAIL exit code so this can gate a deploy
rather than just print numbers for a human to eyeball.

Posts each query fresh (new session per query, KV cache purged first) to
the live /ask-stream and checks the FIRST-turn reply only for the
agent-handoff offer marker ("want me to connect you with" — the literal
fragment api.py's offer-text f-string always contains, see api.py's
_routing_decision block). Deliberately does NOT reply "yes" to any offer
— doing so would drive a real travel_bot conversation forward (real
DB rows, and far enough into the quote flow, a real outbound email/lead)
for what should be a read-only routing check. See "Test Queries Trigger
Real Emails" precedent for why that matters.

Four sets (agent_routing_corpus.json):
  * known_travel        — genuine quote/purchase-intent phrasings; expect
                           the offer to fire (agent_name="ava").
  * travel_info_not_ava — informational travel-insurance questions with NO
                           purchase intent ("What is travel insurance?",
                           "does it cover trip cancellation?"); expect NO
                           offer. This is the regression check for the
                           real bug this corpus was extended for: Ava's
                           first cut scoped on TOPIC ("travel insurance")
                           rather than INTENT (quote/buy), so any
                           travel-insurance-flavored RAG question hijacked
                           the handoff. HARD FAIL class, same as
                           clean_control — only the quote/bind/issue flow
                           is implemented today, so a purely informational
                           question must never be routed away from Layla's
                           own knowledge base.
  * clean_control        — one real query per existing policy type + a
                            greeting; expect NO offer. Any hit here is a
                            false-positive route that would hijack an
                            ordinary Layla query.
  * adversarial           — travel-adjacent wording that is NOT travel
                             insurance; expect NO offer, but via the
                             noisier LLM-fallback band, so treated as WARN
                             not HARD FAIL (see below).

  PASS/FAIL criteria (2026-07-24, updated after the quote-vs-informational
  rescoping — mirrors contamination_corpus_runner.py's discipline):
    - HARD FAIL: any clean_control OR travel_info_not_ava false positive.
      Same invariant as the contamination runner's clean-control rule — an
      ordinary, unambiguous query (including a purely informational
      question about an agent's own topic) must never get hijacked into
      an agent offer.
    - WARN (exits 0): any adversarial false positive, or any known_travel
      case that fails to trigger the offer (recall miss) — both flow
      through the ambiguous LLM-fallback band, which this project has
      already established isn't 100% reliable (see agent_router/core.py's
      threshold-calibration notes).

Usage:
  python3 agent_routing_corpus_runner.py                 # full corpus
  python3 agent_routing_corpus_runner.py --set known_travel
  python3 agent_routing_corpus_runner.py --repeats 3      # N samples/case
  python3 agent_routing_corpus_runner.py --out results.json
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
OFFER_MARKER = "want me to connect you with"

_HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_HERE, "agent_routing_corpus.json")


def _docker(cmd: str) -> None:
    try:
        subprocess.run(
            ["docker", "exec", CONTAINER, "sh", "-c", cmd],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        print(f"  [warn] docker exec failed: {exc}", file=sys.stderr)


def _reset_state() -> None:
    _docker(f"rm -f {KV_CACHE_PATH}")


def _ask_first_turn(query: str, session_id: str, timeout: int = 90) -> str:
    """POST to /ask-stream, return the raw streamed body (metadata blob
    included is fine — we only substring-search for the offer marker,
    which never appears in the trailing JSON)."""
    body = json.dumps({"question": query, "session_id": session_id}).encode()
    req = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def run_case(case: dict, repeats: int) -> dict:
    hits = 0
    for r in range(repeats):
        session = f"routing-corpus-{case['id']}-{r}-{int(time.time())}"
        _reset_state()
        try:
            answer = _ask_first_turn(case["query"], session)
        except Exception as exc:
            print(f"  [warn] request failed for {case['id']}: {exc}", file=sys.stderr)
            continue
        if OFFER_MARKER in answer.lower():
            hits += 1
    return {"id": case["id"], "query": case["query"], "repeats": repeats, "offer_count": hits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["known_travel", "travel_info_not_ava", "clean_control", "adversarial"], default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    sets = [args.set] if args.set else list(corpus.keys())
    all_results = {}
    for set_name in sets:
        cases = corpus[set_name]
        print(f"\n=== {set_name} ({len(cases)} cases x{args.repeats}) ===")
        results = []
        for case in cases:
            r = run_case(case, args.repeats)
            results.append(r)
            rate = r["offer_count"] / r["repeats"]
            print(f"  {r['id']:<20} offer_rate={rate:.0%}  ({r['query']})")
        all_results[set_name] = results

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {args.out}")

    # ── PASS/FAIL verdict ────────────────────────────────────────────────
    hard_fail = False
    warn = False

    for hard_fail_set in ("clean_control", "travel_info_not_ava"):
        if hard_fail_set not in all_results:
            continue
        total = sum(r["repeats"] for r in all_results[hard_fail_set])
        hits = sum(r["offer_count"] for r in all_results[hard_fail_set])
        rate = (hits / total * 100) if total else 0.0
        print(f"\n{hard_fail_set} false-positive rate: {rate:.1f}% ({hits}/{total})")
        if hits > 0:
            hard_fail = True
            print(f"  HARD FAIL: a {hard_fail_set} query triggered the agent-handoff offer.")

    if "adversarial" in all_results:
        total = sum(r["repeats"] for r in all_results["adversarial"])
        hits = sum(r["offer_count"] for r in all_results["adversarial"])
        rate = (hits / total * 100) if total else 0.0
        print(f"adversarial false-positive rate: {rate:.1f}% ({hits}/{total})")
        if hits > 0:
            warn = True
            print("  WARN: an adversarial query triggered the offer (ambiguous LLM-fallback band).")

    if "known_travel" in all_results:
        total = sum(r["repeats"] for r in all_results["known_travel"])
        hits = sum(r["offer_count"] for r in all_results["known_travel"])
        rate = (hits / total * 100) if total else 0.0
        print(f"known_travel recall rate: {rate:.1f}% ({hits}/{total})")
        if hits < total:
            warn = True
            print("  WARN: at least one known-travel query failed to trigger the offer (recall miss).")

    if hard_fail:
        print("\nVERDICT: FAIL")
        sys.exit(1)
    elif warn:
        print("\nVERDICT: PASS (with warnings)")
        sys.exit(0)
    else:
        print("\nVERDICT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
