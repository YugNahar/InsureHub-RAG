"""
Mines per-policy-type DISTINCTIVE VOCABULARY straight out of the knowledge
base, so cross-topic contamination detection stops depending on a
hand-maintained list.

Why this exists (plan_dynamic_contamination_coverage.md, step D1):

  * `_TYPE_GIVEAWAY_TERMS` (multi_source_rag.py) is hand-written and covers
    only 6 names, 2 of which ("fidelity", "transit") aren't even policy_type
    values present in this KB — while the KB actually classifies chunks into
    12 types. Eight of them (commercial, fire, general, home, liability,
    life, personal_accident, travel) have NO giveaway coverage at all.

  * `_text_has_giveaway_contamination` additionally refuses to fire unless
    the offending term is already present in the RETRIEVED context. That
    makes it structurally blind to vocabulary the model produced from its
    own parametric knowledge — confirmed live by the "discharge summaries"
    case, a phrase appearing in ZERO chunks anywhere in this KB.

This module addresses both: the vocabulary is DERIVED (new documents bring
new jargon with no code change) and the resulting check is meant to be run
against the generated ANSWER, independent of what was retrieved.

Scoring is deliberately boring and inspectable — document frequency inside
a type versus outside it — not an embedding model. A human has to be able
to read the output and say "yes, those are giveaways for that product",
which is the D1 go/no-go gate in the plan.

CLI (offline build / review):
    python -m type_vocab_miner --top 30
    python -m type_vocab_miner --out type_vocab.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_META_PATH = os.path.join(
    _HERE, "turbovec_data", "documents", "insurance_docs_meta.ndjson"
)
DEFAULT_OUT_PATH = os.path.join(_HERE, "turbovec_data", "type_vocab.json")

# "general" is genuinely cross-topic by construction — it's the bucket for
# chunks that describe insurance as a whole. Mining it would turn ordinary
# vocabulary ("premium", "policyholder") into a giveaway for a type. It is
# excluded as a MINING TARGET but deliberately kept in the background
# corpus, so a term common in general text is correctly penalised.
EXCLUDED_AS_TARGET = {"general", None, ""}

# Terms shorter than this are noise ("the", "of") or OCR debris.
_MIN_TERM_CHARS = 4

# A term must appear in at least this many DISTINCT chunks of a type before
# it can be that type's giveaway. Guards against a single OCR artifact or
# one-off typo becoming a permanent signal — this KB is known to contain
# extraction corruption (project_pdf_text_extraction_corruption).
MIN_DF_IN_TYPE = 3

# ...and in at least this share of that type's chunks, so a term that is
# rare everywhere (including its own type) doesn't qualify on absolute
# count alone in a large type.
MIN_RATE_IN_TYPE = 0.04

# How much more concentrated in this type than in the rest of the corpus.
# Calibrated 2026-07-28 against a labeled set (21 known-good giveaways drawn
# from the validated hand list, 8 known-junk terms observed in the first D1
# output), NOT guessed:
#
#     thresh   good kept   junk kept
#          8      21/21        8/8     <- original placeholder, no filtering
#         15      21/21        6/8     <- chosen
#         20      20/21        6/8     <- starts destroying real signal
#         30      18/21        5/8
#         50      14/21        2/8
#
# 15 is the knee: it keeps every known-good term while removing the two that
# actually produced false positives in the first probe run ("home" @ 8.94 and
# "insurance covers" @ 10.22, both barely over the old floor). Raising it
# further trades away genuine giveaways for little junk reduction.
#
# The junk that survives at 15 ("finland", "hotels", "policy form") is a
# CORPUS property, not a scoring failure — "finland" genuinely appears only
# in travel documents because this KB's travel guide is from a Finnish
# insurer. No frequency-based measure can tell a country name from product
# jargon; MIN_FOREIGN_HITS is the second line of defence there.
MIN_DISTINCTIVENESS = 15.0

# A type only enters the map once the KB yields enough distinctive terms for
# it. Measured 2026-07-28: life 224, health 169, motor 164, travel 160,
# liability 35, marine 22, crop 17 — versus fire 2, home 2, commercial 0,
# personal_accident 0. The sparse types don't just yield few terms, they
# yield WRONG ones (home -> "machinery", which is engineering/fire
# vocabulary; fire -> "subrogation", a general insurance concept), because
# 1-6 chunks can't separate a product's own jargon from its neighbours'.
#
# Deliberately a THRESHOLD, not a hardcoded list of the 7 types that pass
# today: as KB content is added for fire/home/commercial/personal_accident,
# they cross this bar and join the map with no code change. That is the
# "dynamic" property this whole module exists for.
MIN_TERMS_FOR_TYPE = 10

# Cap per type so the runtime check stays a small, fast membership test and
# stays anchored on the strongest signals rather than the long tail.
MAX_TERMS_PER_TYPE = 150

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Page furniture from the source PDFs. These are course/diploma materials
# whose headers and footers repeat the topic name on every page ("MODULE - 4
# / Practice of General Insurance / Notes / Motor Insurance"), so terms like
# "notes motor", "lesson life" and "documents diploma" correlate PERFECTLY
# with a type and score at the very top — while being layout artifacts, not
# insurance vocabulary. They would "work" as classifiers for entirely the
# wrong reason and would break the moment a document with different
# furniture is ingested. Dropped before scoring.
_BOILERPLATE_WORDS = frozenset("""
    notes note lesson lessons module modules diploma diplomas chapter
    chapters page pages section sections unit units figure table annexure
    appendix contents introduction objectives summary exercise exercises
    practice services institute course
    """.split())

# Structural/boilerplate words that survive the length filter but carry no
# topical signal. Kept short on purpose: the distinctiveness ratio is meant
# to do the real work, and a long hand-tuned stoplist would reintroduce
# exactly the manual maintenance this module exists to remove.
_STOPWORDS = frozenset("""
    shall which their there where when what will would could should about
    under over into from with without that this these those been being have
    has had also such other than then they them his her its our your you
    are was were not but for and the any all may can must upon per each
    same only more most less least very much many some both either neither
    """.split())


def _terms(text: str) -> List[str]:
    """Unigrams + bigrams, normalized. Bigrams matter: the real giveaways in
    this domain are phrases ("bill of lading", "domiciliary hospitalization",
    "discharge voucher"), and a unigram-only miner surfaces their component
    words instead, which are far less distinctive."""
    words = [w for w in _WORD_RE.findall((text or "").lower())]
    keep = [
        w for w in words
        if len(w) >= _MIN_TERM_CHARS
        and w not in _STOPWORDS
        and w not in _BOILERPLATE_WORDS
    ]
    out = list(keep)
    # Bigrams from the RAW word sequence (not the stopword-filtered one) so
    # "bill of lading" survives as a phrase; then drop bigrams that are
    # entirely stopwords/too short.
    for a, b in zip(words, words[1:]):
        if len(a) < 3 or len(b) < 3:
            continue
        # EITHER half being a stopword makes the bigram an article/filler
        # pairing rather than a phrase — "the tpa", "and officers", "this
        # clause", "fall into" all scored highly while carrying no more
        # signal than their content word alone (which is already mined as a
        # unigram). Real domain phrases ("nursing home", "product
        # liability", "commercial vehicles", "private cars") are unaffected.
        if a in _STOPWORDS or b in _STOPWORDS:
            continue
        # Either half being page furniture poisons the bigram ("notes
        # motor", "lesson life", "documents diploma").
        if a in _BOILERPLATE_WORDS or b in _BOILERPLATE_WORDS:
            continue
        out.append(f"{a} {b}")
    return out


def load_chunks(meta_path: str = DEFAULT_META_PATH) -> List[Tuple[str, str]]:
    """-> [(policy_type, text)]"""
    rows: List[Tuple[str, str]] = []
    with open(meta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ptype = (d.get("metadata") or {}).get("policy_type")
            text = d.get("text") or ""
            if text:
                rows.append((ptype, text))
    return rows


def mine(rows: Iterable[Tuple[str, str]]) -> Dict[str, List[dict]]:
    """Document-frequency based distinctiveness, per policy_type.

    For term t and type T:
        rate_in    = df_T(t)     / n_T
        rate_out   = df_other(t) / n_other      (other INCLUDES general)
        score      = rate_in / (rate_out + eps)

    A term qualifies for T when it clears the absolute floor, the in-type
    rate floor, and the distinctiveness ratio. Deliberately NOT a single
    blended score: three independent floors are easier to reason about and
    to tune one at a time, and each one exists to kill a specific failure
    mode documented in the plan.
    """
    by_type_df: Dict[str, Counter] = defaultdict(Counter)
    n_by_type: Counter = Counter()

    for ptype, text in rows:
        key = ptype if ptype else "general"
        n_by_type[key] += 1
        # set() -> DOCUMENT frequency, not term frequency: one chunk
        # repeating a word 30x must not outweigh 30 chunks mentioning it
        # once. Repetition is common in this KB's tabular/OCR'd pages.
        for t in set(_terms(text)):
            by_type_df[key][t] += 1

    total_docs = sum(n_by_type.values())
    eps = 1.0 / max(total_docs, 1)

    # global df across every type, so we can derive "outside this type"
    global_df: Counter = Counter()
    for key, counter in by_type_df.items():
        for t, c in counter.items():
            global_df[t] += c

    results: Dict[str, List[dict]] = {}
    for ptype, counter in by_type_df.items():
        if ptype in EXCLUDED_AS_TARGET:
            continue
        n_T = n_by_type[ptype]
        n_other = total_docs - n_T
        if n_T <= 0 or n_other <= 0:
            continue

        scored: List[dict] = []
        for term, df_T in counter.items():
            if df_T < MIN_DF_IN_TYPE:
                continue
            rate_in = df_T / n_T
            if rate_in < MIN_RATE_IN_TYPE:
                continue
            df_out = global_df[term] - df_T
            rate_out = df_out / n_other
            score = rate_in / (rate_out + eps)
            if score < MIN_DISTINCTIVENESS:
                continue
            scored.append({
                "term": term,
                "df_in": df_T,
                "n_type": n_T,
                "df_out": df_out,
                "rate_in": round(rate_in, 4),
                "rate_out": round(rate_out, 5),
                "score": round(score, 2),
            })
        scored.sort(key=lambda r: (-r["score"], -r["df_in"], r["term"]))
        results[ptype] = scored
    return results


def build_map(rows: Iterable[Tuple[str, str]] | None = None,
              meta_path: str = DEFAULT_META_PATH) -> Dict[str, List[str]]:
    """The artifact the runtime check consumes: {policy_type: [term, ...]}.

    Applies the viability threshold, so a type with too little KB content to
    yield trustworthy vocabulary is simply absent from the map — the runtime
    check then has nothing to say about it, which is the correct behaviour
    (silence beats a confident wrong flag from 2 chunks).
    """
    if rows is None:
        rows = load_chunks(meta_path)
    mined = mine(rows)
    out: Dict[str, List[str]] = {}
    for ptype, scored in mined.items():
        if len(scored) < MIN_TERMS_FOR_TYPE:
            continue
        out[ptype] = [r["term"] for r in scored[:MAX_TERMS_PER_TYPE]]
    return out


def write_map(path: str = DEFAULT_OUT_PATH, meta_path: str = DEFAULT_META_PATH) -> Dict[str, List[str]]:
    m = build_map(meta_path=meta_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(m, fh, indent=2)
    return m


# ── Runtime side (D2: LOG-ONLY, drops nothing) ───────────────────────────
# Loaded once per process. Absent/corrupt file -> empty map -> the check is
# a silent no-op, which is the required fail-open behaviour.
_RUNTIME_MAP: Dict[str, frozenset] | None = None

# A unit must contain at least this many DISTINCT foreign-type terms before
# it is even reported. One weak term is exactly how
# project_candidate_vocab_runaway_matching_bug mislabelled 202/203 chunks —
# a single-hit trigger is known-unsafe in this codebase.
MIN_FOREIGN_HITS = 2


def _load_runtime_map(path: str = DEFAULT_OUT_PATH) -> Dict[str, frozenset]:
    global _RUNTIME_MAP
    if _RUNTIME_MAP is not None:
        return _RUNTIME_MAP
    try:
        with open(path) as fh:
            raw = json.load(fh)
        _RUNTIME_MAP = {k: frozenset(v) for k, v in raw.items() if v}
    except Exception:
        _RUNTIME_MAP = {}
    return _RUNTIME_MAP


def foreign_type_hits(text: str, query_policy_type: str) -> Dict[str, List[str]]:
    """Which OTHER policy types' distinctive vocabulary appears in *text*.

    Deliberately does NOT consult the retrieved context — that is the whole
    point. `_text_has_giveaway_contamination` requires the term to be
    present in what was retrieved and is therefore structurally blind to
    vocabulary the model produced from its own parametric knowledge (the
    "discharge summaries" class). This asks a different question: is this
    vocabulary characteristic of a DIFFERENT product? That has an answer
    whether or not the term was retrieved.

    Returns {other_type: [matched terms]} for types clearing MIN_FOREIGN_HITS.
    Caller applies exemptions; this function only reports.
    """
    vocab = _load_runtime_map()
    if not vocab or not text:
        return {}
    present = set(_terms(text))
    if not present:
        return {}
    out: Dict[str, List[str]] = {}
    for ptype, terms in vocab.items():
        if ptype == query_policy_type:
            continue
        hits = sorted(present & terms)
        if len(hits) >= MIN_FOREIGN_HITS:
            out[ptype] = hits
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=DEFAULT_META_PATH)
    ap.add_argument("--top", type=int, default=30, help="terms per type to print")
    ap.add_argument("--out", help="write full JSON map here")
    args = ap.parse_args()

    rows = load_chunks(args.meta)
    print(f"loaded {len(rows)} chunks")
    counts = Counter(p if p else "general" for p, _ in rows)
    print("chunks per type:", dict(sorted(counts.items())))
    print()

    results = mine(rows)
    for ptype in sorted(results):
        terms = results[ptype]
        print(f"=== {ptype}  ({counts[ptype]} chunks, {len(terms)} distinctive terms) ===")
        for r in terms[: args.top]:
            print(f"   {r['score']:8.1f}  df_in={r['df_in']:<4} df_out={r['df_out']:<4} {r['term']}")
        if not terms:
            print("   (none cleared the floors)")
        print()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
