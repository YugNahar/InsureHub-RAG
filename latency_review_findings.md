# Opus review — latency work vs. plan_latency.md

**Reviewer:** Opus 4.8 · **Implementer:** Sonnet · **Commits reviewed:** `667d5b8`, `018adb1`, `90e657f`, `6f1d66e`

**Verdict: not yet done.** The code is directionally right and matches the plan's design, but **none of the latency work has ever actually executed**, and the measurement harness has defects that would produce a silently-empty or wrong baseline. Fix the blockers before trusting any number.

---

## What is correct (verified, no action needed)

- **Device auto-detect logic** (`turbovec_store.py`) — both named load sites get `device=`, `MODEL_DEVICE` override works, defaults to auto. Verified live: both models load *and run real inference* with `device="cpu"` on this host. Correctly a no-op on CPU.
- **Compressor inherits the shared embedder** — confirmed both `ContextCompressor(...)` sites (`multi_source_rag.py:3496`, `rag.py:604`) pass `vector_store.embed_model`, i.e. the shared getter's instance. The plan's fact-E claim ("compressor gets GPU for free, no separate change") **holds**.
- **TTFT placement is semantically right** — set once on the first *content* token, and the buffered/non-streaming fallback path is covered.
- **Runner conventions** — mirrors `contamination_corpus_runner.py`, checks `DISABLE_QUERY_CACHE` and warns loudly. Good.
- **Contamination Phase 2 isolation redesign is live and behaving as designed** — `drop=1/8 (gap=5.24x)`, `drop=2/8 (gap=14.83x)`: genuine minorities, versus the old ratio-to-max design's 6/8 false positive.

---

## BLOCKERS — fix before any baseline is trusted

### B1. The TTFT + GPU code has never actually run
The container started `2026-07-22T12:10:14Z` (17:40 IST); commit `90e657f` landed 17:54 IST. Live TIMING lines carry **no `ttft=` field**. The code is on disk inside the container (bind mount) but was never loaded by the running process — it is **compile-checked only, never observed working**. This violates the plan's own "verify live" discipline, and the entire Phase 0 premise depends on `ttft=` existing.
**Fix:** restart `insurehub_api`, then confirm a real `ttft=` value appears in a TIMING line *and* that `[SharedModels] ... (device=cpu)` logs on model load.

### B2. The runner requires `ttft=` and will match zero lines against un-reloaded code
`_TIMING_RE` has `ttft=(\S+)` as a **mandatory** group. Against the currently-running process every case returns `{}` → the table prints all `?` with no error. Running the harness right now yields a **silently empty baseline that looks like a successful run**.
**Fix:** make the harness fail loudly when it matches zero TIMING lines (or make `ttft=` optional and report it as missing), so an un-deployed build can never masquerade as a result.

---

## DEFECTS — would produce wrong or missing data

### D1. The follow-up case can never match its TIMING line
The runner matches by `query[:40] in line`, but the TIMING line logs **`retrieval_query`**, which is *rewritten* for follow-ups (`multi_source_rag.py:4353-4379` merge/reformulate; `_correct_typos` at `:4464` runs on every query). For the follow-up case the measured text (`"What's excluded under it?"`) never appears in the log line — the rewritten standalone query does.
**Fix:** log the original question (or a per-request id) alongside `retrieval_query`, and match on that. The runner already generates a unique `session_id` — logging and matching on it is the cleanest fix, and also solves D4.

### D2. The refusal case can never produce a TIMING line at all
Gated refusals `yield ... ; return` **before** reaching the TIMING log (e.g. the reranker-gate refusal). Confirmed empirically earlier this session: a refusal query emitted no TIMING line. The plan asked for a refusal case in the 6-query set, but the instrumentation deliberately skips early returns (documented at `:3974`). As written this row is **permanently blank**.
**Fix:** either instrument the early-return paths (at minimum total + ttft), or drop the refusal case from the set and document why. Do not leave a case that silently never reports.

### D3. Follow-up case reuses one `session_id` across cold and warm passes
The warm pass therefore runs as turn 3–4 with the cold pass's history already in the session, not as a clean 2-turn conversation. Cold and warm are measuring **different conversational states**.
**Fix:** fresh `session_id` per pass.

### D4. Log-window fragility
`docker logs --tail 60` may not contain the target TIMING line on a busy container — each request emits several INFO lines (Compressor, agent_hub, TIMING).
**Fix:** widen the window, and match on the unique `session_id` (see D1) rather than query text.

---

## GAPS vs. the plan (incomplete, not wrong)

### G1. Phase 1a is code-only — the infra half is missing
The plan's Phase 1a explicitly requires the image to ship **CUDA-enabled torch** and the runtime to **expose the GPU** (`--gpus all` / compose `deploy.resources.reservations.devices`). Neither `Dockerfile` nor `docker-compose.yml` was touched. Without that, `torch.cuda.is_available()` stays `False` on the GPU server and the whole change is a **silent no-op there** — it will look deployed and do nothing.
**Fix:** land the image/compose GPU config. Phase 1a is not done until this exists and `device=cuda` is observed in the server's logs.

### G2. Third model-load site missed
`semantic_chunker.py:56` builds `SentenceTransformer(EMBED_MODEL_NAME)` with no `device=`, bypassing both the shared getter and the device fix. It is a lazy fallback (fires only when no `embed_model` is passed) and sits on the **ingestion** path, not the answer-latency path — so it does not affect the 5–6s goal — but it stays CPU-bound on a GPU host and holds a duplicate model copy in memory.
**Fix:** pass the shared model / device. Low priority, but the plan's Phase 1a said "the two shared-model load sites" — that enumeration was itself incomplete.

---

## MINOR

- **M1.** `_ms()` returns `None` for `"n/a"`; the table does `w.get('total_ms', '?')`, so a present-but-`None` value prints `None` rather than `?`. Cosmetic.
- **M2.** `_purge_kv_cache()` is redundant when `DISABLE_QUERY_CACHE=1`. Harmless belt-and-braces.

---

## PROCESS

- **P1.** Commit `90e657f` bundles **two unrelated plans** — latency TTFT instrumentation *and* the contamination Phase 2 statistic redesign. The plan's own principle is "one change at a time, re-measured"; bundling makes attribution and rollback harder. Not worth rewriting history; worth not repeating.
- **P2.** The Phase 2 corpus sweep **never completed** (stopped, no `phase2_isolation_gate_sweep.json` written). The isolation-gate redesign's broad validation is still outstanding, even though live logs show it behaving sensibly. Re-run before considering contamination Phase 2 settled.

---

## Recommended order for Sonnet

1. **B1** — restart, confirm `ttft=` and `device=` actually appear. Nothing else is measurable until this is true.
2. **B2, D1, D2, D3, D4** — fix the harness (session-id matching + loud failure) so a baseline can't be silently empty or wrong.
3. Run the **real Phase 0 baseline** on an idle backend, commit the table.
4. **G1** — GPU image/compose config, then re-measure on the server (the only place the GPU win is visible).
5. **G2, M1, M2** — cleanup.
6. **P2** — re-run the contamination Phase 2 sweep.
