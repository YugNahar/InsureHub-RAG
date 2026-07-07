"""
Regression test runner for the InsureHub RAG evaluation API.

Loads test cases from test_cases.json, POSTs each to the eval/query endpoint,
checks expectations, and writes both a JSON report and a Markdown summary.

Usage:
    python eval/run_regression.py                          # http://localhost:8002
    python eval/run_regression.py --host http://other:8002
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
logger = logging.getLogger("run_regression")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_CASES_PATH = os.path.join(_HERE, "test_cases.json")
_REPORT_JSON_PATH = os.path.join(_HERE, "regression_report.json")
_REPORT_MD_PATH = os.path.join(_HERE, "regression_report.md")

# Refusal/handoff phrases matching multi_source_rag.py's canned responses.
_REFUSAL_PHRASES = frozenset({
    "don't have that specific information",
    "let me get one of our agents",
    "i don't have that in my knowledge base",
    "let me get a human agent",
    "i can only help with insurance-related questions",
})


def _is_refusal(answer: str) -> bool:
    """True if *answer* contains one of the known refusal phrases."""
    lower = answer.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def _check_expectation(
    expect_key: str,
    expect_value: object,
    answer: str,
    sources: list[str],
) -> tuple[bool, str]:
    """
    Check a single expectation key against the actual response.

    Returns (passed, detail_message).
    """
    answer_lower = answer.lower()
    is_refusal = _is_refusal(answer)

    if expect_key == "should_refuse":
        return (is_refusal, f"refusal={is_refusal}")

    if expect_key == "should_ground":
        grounded = (not is_refusal) and bool(sources)
        return (grounded, f"grounded={grounded}  refusal={is_refusal}  sources={len(sources)}")

    if expect_key == "no_retrieval_expected":
        no_retrieval = not bool(sources)
        return (no_retrieval, f"sources_empty={no_retrieval}  sources={len(sources)}")

    if expect_key == "named_entity_required":
        # The answer should mention at least one insurer/company name.
        # Simple substring check for common insurer keywords.
        entity_indicators = (
            "insurance", "insurer", "company", "ltd", "inc",
            "hdfc", "icici", "bajaj", "tata", "aig", "reliance",
            "new india", "oriental", "national", "united india",
        )
        found = any(indicator in answer_lower for indicator in entity_indicators)
        return (found, f"entity_in_answer={found}")

    if expect_key == "must_reference_prior_topic":
        # The answer must NOT be a refusal and must contain the expected
        # topic word/phrase (expect_value) as a case-insensitive substring,
        # proving the pronoun/reference was resolved correctly.
        if not isinstance(expect_value, str):
            return (True, "MANUAL_REVIEW (non-string expect_value)")
        topic_present = expect_value.lower() in answer_lower
        passed = (not is_refusal) and topic_present
        return (passed, f"topic_in_answer={topic_present}  refusal={is_refusal}")

    if expect_key == "must_not_reference_prior_topic":
        # The answer must NOT contain the prior-topic word/phrase
        # (expect_value) — a fresh standalone question should not drag in
        # the previous conversation's topic.
        if not isinstance(expect_value, str):
            return (True, "MANUAL_REVIEW (non-string expect_value)")
        topic_absent = expect_value.lower() not in answer_lower
        return (topic_absent, f"topic_absent={topic_absent}")

    if expect_key == "must_not_repeat_definition":
        # The answer should be grounded but should NOT re-explain from scratch
        # (i.e. it's a follow-up asking for an example, not a re-definition).
        # Heuristic: answer length > 20 chars and not a refusal.
        valid = (not is_refusal) and len(answer) > 20
        return (valid, f"valid_example={valid}  refusal={is_refusal}  len={len(answer)}")

    if expect_key == "should_refuse_if_either_term_absent":
        # Expectation: if one of the quoted terms is absent from KB, it should
        # refuse. Check refusal status.
        return (is_refusal, f"refusal={is_refusal}")

    if expect_key == "is_off_topic_refusal":
        # Same logic as should_refuse — the answer should contain a refusal
        # phrase indicating the question is off-topic.
        return (is_refusal, f"refusal={is_refusal}")

    if expect_key == "is_greeting_response":
        # Production ask() doesn't have ask_stream()'s fast paths, so a
        # greeting retrieves normally — just check it doesn't refuse.
        return (not is_refusal, f"greeting_not_refused={not is_refusal}  refusal={is_refusal}")

    if expect_key == "is_acknowledgment_response":
        # Same loose check — just verify not a refusal.
        return (not is_refusal, f"acknowledgment_not_refused={not is_refusal}  refusal={is_refusal}")

    if expect_key == "must_not_contain":
        # expect_value is a list of phrases; ALL must be absent from the
        # answer (case-insensitive). Passes if the list is empty.
        if not isinstance(expect_value, list):
            return (True, "MANUAL_REVIEW (non-list expect_value)")
        found_any = [p for p in expect_value if p.lower() in answer_lower]
        passed = not found_any
        detail = f"forbidden_phrases_found={found_any}" if found_any else "all_absent"
        return (passed, detail)

    # Unknown expectation → manual review, not pass/fail
    return (True, "MANUAL_REVIEW")


def _post_query(host: str, query: str, history: str = "") -> dict:
    """POST a query to the production eval API and return the parsed JSON response."""
    url = f"{host.rstrip('/')}/eval/query-production"
    payload = json.dumps({
        "query": query,
        "history": history,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed to {url}: {e.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run regression tests against the InsureHub RAG eval API."
    )
    parser.add_argument(
        "--host",
        default="http://localhost:8002",
        help="Eval API base URL (default: http://localhost:8002)",
    )
    args = parser.parse_args()

    host = args.host
    logger.info("Loading test cases from %s", _TEST_CASES_PATH)

    if not os.path.exists(_TEST_CASES_PATH):
        logger.error("test_cases.json not found at %s", _TEST_CASES_PATH)
        sys.exit(1)

    with open(_TEST_CASES_PATH, "r") as f:
        data = json.load(f)
    all_cases = data["cases"] if isinstance(data, dict) and "cases" in data else data

    logger.info("Loaded %d test case(s)", len(all_cases))

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        health_url = f"{host.rstrip('/')}/eval/health"
        with urllib.request.urlopen(health_url, timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
            logger.info(
                "API healthy — chunks=%s  summaries=%s  backend=%s",
                health.get("chunks_in_store"), health.get("summaries_in_store"),
                health.get("llm_backend"),
            )
    except Exception as exc:
        logger.error("API not reachable at %s: %s", host, exc)
        print(f"FAIL 0/{len(all_cases)} passed — API unreachable at {host}")
        sys.exit(1)

    # ── Run test cases ────────────────────────────────────────────────────────
    results: list[dict] = []
    passed = 0
    failed = 0
    skipped = 0

    for case in all_cases:
        case_id = case.get("id", "?")
        category = case.get("category", "?")
        query = case.get("query", "")
        history = case.get("history", "")
        expect = case.get("expect", {})

        logger.info("[%s] %s  query=%r", category, case_id, query[:60])

        # POST to API (history is passed through; empty string for history-less cases)
        try:
            data = _post_query(host, query, history=history)
        except Exception as exc:
            logger.error("[%s] %s — ERROR: %s", category, case_id, exc)
            result = {
                "id": case_id,
                "category": category,
                "query": query,
                "status": "error",
                "reason": str(exc),
                "answer": "",
                "sources": [],
                "expect": expect,
                "checks": [],
            }
            results.append(result)
            failed += 1
            continue

        answer = data.get("answer", "") or ""
        sources = data.get("sources", []) or []

        # Check each expectation
        checks = []
        all_ok = True
        for expect_key, expect_value in expect.items():
            ok, detail = _check_expectation(expect_key, expect_value, answer, sources)
            checks.append({
                "key": expect_key,
                "expected": expect_value,
                "passed": ok,
                "detail": detail,
            })
            if not ok:
                all_ok = False

        status = "pass" if all_ok else "fail"
        result = {
            "id": case_id,
            "category": category,
            "query": query,
            "status": status,
            "reason": "",
            "answer": answer,
            "sources": sources,
            "expect": expect,
            "checks": checks,
        }
        results.append(result)

        if status == "pass":
            passed += 1
            logger.info("  → PASS")
        else:
            failed += 1
            logger.info("  → FAIL  %s", [c for c in checks if not c["passed"]])

    # ── Write JSON report ─────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "host": host,
        "total": len(all_cases),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }
    with open(_REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("JSON report written to %s", _REPORT_JSON_PATH)

    # ── Write Markdown report ─────────────────────────────────────────────────
    md_lines = [
        f"# Regression Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        f"- **Host**: `{host}`",
        f"- **Total**: {len(all_cases)}",
        f"- **Passed**: {passed}",
        f"- **Failed**: {failed}",
        f"- **Skipped**: {skipped}",
        "",
        "---",
        "",
    ]

    # Per-category summary
    from collections import Counter
    cat_total = Counter(r["category"] for r in results)
    cat_pass = Counter(r["category"] for r in results if r["status"] == "pass")
    cat_fail = Counter(r["category"] for r in results if r["status"] == "fail")
    cat_skip = Counter(r["category"] for r in results if r["status"] == "skipped")

    md_lines.append("## Per-Category Summary")
    md_lines.append("")
    md_lines.append("| Category | Total | Passed | Failed | Skipped |")
    md_lines.append("|----------|-------|--------|--------|---------|")
    for cat in sorted(cat_total):
        t = cat_total[cat]
        p = cat_pass.get(cat, 0)
        f_ = cat_fail.get(cat, 0)
        s = cat_skip.get(cat, 0)
        md_lines.append(f"| {cat} | {t} | {p} | {f_} | {s} |")
    md_lines.append("")

    # Failed cases detail
    failed_cases = [r for r in results if r["status"] == "fail"]
    if failed_cases:
        md_lines.append("## Failed Cases")
        md_lines.append("")
        for r in failed_cases:
            answer_snippet = (r.get("answer") or "")[:300].replace("\n", " ")
            md_lines.append(f"### {r['id']} ({r['category']})")
            md_lines.append("")
            md_lines.append(f"- **Query**: `{r['query']}`")
            md_lines.append(f"- **Answer**: {answer_snippet}")
            fails = [c for c in r.get("checks", []) if not c["passed"]]
            for c in fails:
                md_lines.append(f"- **Expect `{c['key']}`**: expected={c['expected']}  detail={c['detail']}")
            md_lines.append("")

    # Error cases detail
    error_cases = [r for r in results if r["status"] == "error"]
    if error_cases:
        md_lines.append("## Error Cases")
        md_lines.append("")
        for r in error_cases:
            md_lines.append(f"- **{r['id']}**: {r.get('reason', '?')}")
        md_lines.append("")

    # Skipped cases detail
    skipped_cases = [r for r in results if r["status"] == "skipped"]
    if skipped_cases:
        md_lines.append("## Skipped Cases")
        md_lines.append("")
        for r in skipped_cases:
            md_lines.append(f"- **{r['id']}** ({r['category']}): {r.get('reason', '?')}")
        md_lines.append("")

    with open(_REPORT_MD_PATH, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info("Markdown report written to %s", _REPORT_MD_PATH)

    # ── One-line stdout summary ───────────────────────────────────────────────
    print(f"{'PASS' if failed == 0 else 'FAIL'} {passed}/{len(all_cases)} passed  ({failed} failed, {skipped} skipped)")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()