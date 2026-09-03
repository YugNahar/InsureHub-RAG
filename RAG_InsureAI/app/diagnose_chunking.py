"""
One-shot diagnostic for the local-vs-remote chunk-count investigation
(2026-09-03). Run directly wherever it ends up (works on either the
local dev container or the remote server, since it only reads this
machine's own vector store and compares against a fixed local baseline
captured on the dev machine):

    docker exec insurehub_api python3 /app/app/diagnose_chunking.py

Prints, for every document known to exist on both the local dev
machine and this machine: pymupdf4llm's version, this machine's own
max page number and chunk count per document, the local baseline for
the same, and whether they differ. A page-count mismatch on a
document that pymupdf4llm loaded successfully (no fallback warning)
means the underlying PDF content itself differs between the two
machines -- not a code or chunking-pipeline difference, since both
sides are confirmed to be running the same commit.
"""
import sys

sys.path.insert(0, "/app/app")

try:
    import pymupdf4llm
    PYMUPDF4LLM_VERSION = getattr(pymupdf4llm, "__version__", "unknown")
except Exception as exc:
    PYMUPDF4LLM_VERSION = f"IMPORT FAILED: {exc}"

from turbovec_store import TurboVecStore  # noqa: E402

# Captured on the local dev machine 2026-09-03 via the exact same
# max-page/chunk-count computation this script runs below -- the
# reference point every other machine gets compared against.
LOCAL_BASELINE = {
    "InsuranceFundamentals.pdf": {"max_page": 2, "chunks": 10},
    "health_insurance_guide.pdf": {"max_page": 2, "chunks": 22},
    "insurance_terms_glossary.pdf": {"max_page": 5, "chunks": 19},
    "liability_insurance_guide.pdf": {"max_page": 2, "chunks": 21},
    "life_insurance_guide.pdf": {"max_page": 2, "chunks": 13},
    "m3-f2.pdf": {"max_page": 13, "chunks": 20},
    "m4-3f.pdf": {"max_page": 18, "chunks": 52},
    "m4-5f.pdf": {"max_page": 19, "chunks": 37},
    "marine_insurance_guide.pdf": {"max_page": 2, "chunks": 23},
    "motor_insurance_guide.pdf": {"max_page": 3, "chunks": 24},
    "personal_accident_insurance_guide.pdf": {"max_page": 3, "chunks": 22},
    "travel_insurance_guide.pdf": {"max_page": 1, "chunks": 8},
    "travel_insurance_guide_v2.pdf": {"max_page": 3, "chunks": 25},
}


def main():
    print(f"pymupdf4llm version: {PYMUPDF4LLM_VERSION}")
    print()

    store = TurboVecStore(collection_name="insurance_docs", persist_subdir="documents")

    header = f"{'filename':<40} {'local pg':>9} {'here pg':>8} {'local ch':>9} {'here ch':>8}  {'flag'}"
    print(header)
    print("-" * len(header))

    for filename, baseline in LOCAL_BASELINE.items():
        pages = set()
        chunk_count = 0
        for _id, md in store._metadatas.items():
            if md.get("source", "").endswith(filename):
                pages.add(md.get("page"))
                chunk_count += 1
        max_page = max((p for p in pages if p is not None), default=0)

        page_diff = max_page != baseline["max_page"]
        chunk_diff = chunk_count != baseline["chunks"]
        flag = ""
        if chunk_count == 0:
            flag = "<<< NOT FOUND HERE"
        elif page_diff or chunk_diff:
            flag = "<<< DIFFERS"

        print(
            f"{filename[:38]:<40} {baseline['max_page']:>9} {max_page:>8} "
            f"{baseline['chunks']:>9} {chunk_count:>8}  {flag}"
        )

    print()
    print("If a row says DIFFERS with a LOWER page count here than local, and")
    print("pymupdf4llm's version above matches the local machine's, the most")
    print("likely explanation is that a reformatted/corrected version of that")
    print("PDF was uploaded locally and never re-uploaded here -- not a code")
    print("or pipeline difference (both machines confirmed on the same commit).")


if __name__ == "__main__":
    main()
