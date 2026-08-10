#!/usr/bin/env python3
"""
One-off generator for the Phase 0 claims sub-corpus
(plan_claim_answer_correctness.md at the repo root).

Appends two new categories to contamination_corpus.json:

  claims_cross_type_repro  — 36 cases (12 policy types x 3 claim-process
      phrasings). forbidden_phrases per case = every OTHER type's
      multi-word keyword phrases from metadata_tagger.py's
      _POLICY_TYPE_HINTS, keeping only phrases that are NOT shared by two
      or more types (a phrase like "third party liability" or "accidental
      death" that legitimately belongs to more than one type is dropped
      from every list rather than risk a false positive — same discipline
      as the plan's "never a bare single word" guard, generalized to
      "never an ambiguous shared phrase" for auto-generated cases).
      Single-word keywords are excluded entirely (bare-word false-positive
      history: project_motor_bareword_giveaway_generalization,
      project_compound_word_joiner_bug).

  claims_factual_repro     — 3 cases for the exact bad phrasings confirmed
      live and fixed on 2026-08-04 (travel brand leak, motor stolen-
      vehicle-transfer inversion, life 30-day-trigger + third-party-motor
      contamination). Running this category is the regression lock for
      TODAY's fixes; claims_cross_type_repro is the exploratory baseline
      for the other 9 types the plan still needs to address.

This script is meant to be run ONCE to produce the corpus additions, then
deleted or kept for reference — it is not part of the regression-running
path (contamination_corpus_runner.py is).
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_HERE, "contamination_corpus.json")
METADATA_TAGGER_PATH = os.path.join(_HERE, "app", "metadata_tagger.py")

_DISPLAY_NAME = {
    "motor": "motor",
    "health": "health",
    "life": "life",
    "travel": "travel",
    "home": "home",
    "personal_accident": "personal accident",
    "fire": "fire",
    "marine": "marine",
    "liability": "liability",
    "commercial": "commercial",
    "crop": "crop",
    "cyber": "cyber",
}

_PHRASINGS = [
    "how do I claim {name} insurance",
    "what is the {name} claim process",
    "walk me through claiming on a {name} insurance policy",
]


def _extract_keywords_by_type() -> dict:
    text = open(METADATA_TAGGER_PATH).read()
    start = text.find("_POLICY_TYPE_HINTS: dict[str, dict] = {")
    end = text.find("\n_VALID_POLICY_TYPES")
    block = text[start:end]
    type_starts = [
        (m.start(), m.group(1)) for m in re.finditer(r'^    "([a-z_]+)": \{', block, re.MULTILINE)
    ]
    type_starts.append((len(block), None))
    kw_by_type = {}
    for i in range(len(type_starts) - 1):
        start_i, tname = type_starts[i]
        end_i = type_starts[i + 1][0]
        chunk = block[start_i:end_i]
        kw_match = re.search(r'"keywords":\s*\[(.*?)\]', chunk, re.DOTALL)
        if kw_match:
            kws = re.findall(r'"([^"]+)"', kw_match.group(1))
            # multi-word only — bare single words are the documented
            # false-positive source (see module docstring)
            kw_by_type[tname] = [k for k in kws if " " in k or "-" in k]
    return kw_by_type


def _distinctive_phrases_by_type(kw_by_type: dict) -> dict:
    """Drop any phrase shared by 2+ types (e.g. "third party liability" in
    both motor and liability, "accidental death" in both life and
    personal_accident) — a shared phrase can't safely mark ONE type as
    foreign to another."""
    counts = {}
    for kws in kw_by_type.values():
        for k in kws:
            counts[k] = counts.get(k, 0) + 1
    return {t: sorted({k for k in kws if counts[k] == 1}) for t, kws in kw_by_type.items()}


def build_cross_type_cases(distinctive: dict) -> list:
    cases = []
    for own_type, own_name in _DISPLAY_NAME.items():
        # Union of every OTHER type's distinctive phrases.
        forbidden = sorted({
            p for other_type, phrases in distinctive.items()
            if other_type != own_type
            for p in phrases
        })
        for i, template in enumerate(_PHRASINGS, 1):
            cases.append({
                "id": f"claims-crosstype-{own_type}-{i:02d}",
                "category": "claims_cross_type_repro",
                "query": template.format(name=own_name),
                "mode": "detailed",
                "expected_topic": own_type,
                "forbidden_phrases": forbidden,
                "forbidden_topics": [t for t in _DISPLAY_NAME if t != own_type],
                # This type's OWN distinctive vocabulary must never be
                # flagged against itself even though it can't appear in
                # `forbidden` by construction — kept explicit for clarity
                # when a human reads a failing case.
                "must_allow_phrases": distinctive.get(own_type, []),
                "source": "plan_claim_answer_correctness.md Phase 0 (auto-generated 2026-08-04)",
                "notes": (
                    f"Auto-generated claims-sub-corpus case: does a {own_name} "
                    "claim-process answer import another policy type's "
                    "distinctive vocabulary? Baseline for Phase 1 (per-point "
                    "type-attribution gate)."
                ),
            })
    return cases


def build_factual_repro_cases() -> list:
    return [
        {
            "id": "claims-factual-travel-brandleak-01",
            "category": "claims_factual_repro",
            "query": "How to claim the travel insurance?",
            "mode": "detailed",
            "expected_topic": "travel",
            "forbidden_phrases": ["my pages", "if mobile", "if travel insurance"],
            "forbidden_topics": [],
            "must_allow_phrases": [],
            "source": "project_third_party_brand_leak.md (fixed 2026-08-04)",
            "notes": (
                "Confirmed live 2026-08-04: a travel claim answer named a "
                "different insurer's ('If P&C') own portal/app/product as "
                "if it were ours. Fixed via multi_source_rag.py third-party "
                "brand-strip guard + prompt rules. Regression lock."
            ),
        },
        {
            "id": "claims-factual-motor-stolenvehicle-01",
            "category": "claims_factual_repro",
            "query": "Explain the motor insurance claim process in detail",
            "mode": "detailed",
            "expected_topic": "motor",
            "forbidden_phrases": [
                "must be transferred in the name of the insured",
                "transferred in the name of the insured",
            ],
            "forbidden_topics": [],
            "must_allow_phrases": [],
            "source": "project (fixed 2026-08-04)",
            "notes": (
                "Confirmed live 2026-08-04: a motor claim answer inverted "
                "m4-3f.pdf p14 — the source says the insurer asks the RTO "
                "NOT to transfer the stolen vehicle's registration/ownership "
                "(to stop the thief disposing of it); the answer said the "
                "opposite. Fixed via _fc_line_is_always_false_claim. "
                "Regression lock."
            ),
        },
        {
            "id": "claims-factual-life-30day-thirdparty-01",
            "category": "claims_factual_repro",
            "query": "How to claim a life insurance?",
            "mode": "detailed",
            "expected_topic": "life",
            "forbidden_phrases": [
                "from the date of the incident",
                "such as a car accident",
                "settling with them first",
            ],
            "forbidden_topics": ["motor"],
            "must_allow_phrases": [],
            "source": "project (fixed 2026-08-04)",
            "notes": (
                "Confirmed live 2026-08-04 across 3 independent generations: "
                "a life claim answer (a) re-anchored the IRDA Reg 8(3) "
                "30-day deadline to the incident date instead of receipt of "
                "papers, and (b) imported motor's third-party/car-accident "
                "settlement concept with zero grounding in the retrieved "
                "life-insurance sources. Fixed via two new checks in "
                "_fc_line_is_always_false_claim. Regression lock."
            ),
        },
    ]


def main() -> None:
    kw_by_type = _extract_keywords_by_type()
    distinctive = _distinctive_phrases_by_type(kw_by_type)

    print("Distinctive (non-shared) multi-word phrase counts per type:")
    for t, phrases in distinctive.items():
        print(f"  {t}: {len(phrases)} / {len(kw_by_type.get(t, []))} kept")

    cross_type_cases = build_cross_type_cases(distinctive)
    factual_cases = build_factual_repro_cases()

    corpus = json.load(open(CORPUS_PATH))
    existing_ids = {c["id"] for c in corpus["cases"]}
    new_cases = [c for c in cross_type_cases + factual_cases if c["id"] not in existing_ids]
    skipped = len(cross_type_cases) + len(factual_cases) - len(new_cases)

    corpus["cases"].extend(new_cases)
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"\nAdded {len(new_cases)} new case(s) "
          f"({len(cross_type_cases)} cross-type + {len(factual_cases)} factual-repro, "
          f"{skipped} already present, skipped).")
    print(f"Corpus now has {len(corpus['cases'])} total cases.")


if __name__ == "__main__":
    main()
