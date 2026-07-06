async def _contextualize_query(question: str, history: str) -> str:
    """Rewrite the question into a standalone, self-contained form by
    resolving pronouns and implicit references against recent conversation
    history. Runs on every turn — a true first-turn or already-standalone
    question should be returned unchanged, not gated behind a separate
    followup/not-followup classifier.

    *history* is a flat "User: ...\\nAssistant: ..." string as built by
    ConversationAgent._build_history_string(). Uses only the last 1-2
    turns via _split_history_turns().

    Fast-path: if the question contains no reference token at all, it's
    structurally standalone — skip the LLM call entirely. This is a
    latency optimization only; the LLM prompt below is what actually
    enforces correctness (a false-positive regex match just costs one
    extra LLM call that correctly returns the question unchanged).

    Fail-safe: on any exception, timeout, or empty response, return the
    original question unchanged.
    """
    if not history or not history.strip():
        return question

    if not _REFERENCE_TOKENS.search(question.strip().lower()):
        return question

    lines = _split_history_turns(history)
    recent = lines[-4:]
    if not recent:
        return question
    history_text = "\n".join(recent)

    prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"New question: {question}\n\n"
        "Does the new question contain a pronoun or implicit reference "
        "(e.g. 'it', 'that', 'those', 'their', 'the second one') that "
        "depends on the conversation above to be understood?\n"
        "If YES, rewrite the question to resolve that reference, "
        "replacing the pronoun/reference with the specific thing it "
        "refers to. If the reference is to an ordinal position in a "
        "numbered or listed answer above (e.g. 'the second point', "
        "'point 3', 'the last one'), rewrite it to name the SPECIFIC "
        "subject of that one point only — do not fold in neighboring "
        "points.\n"
        "If NO — the question is already a complete, standalone "
        "question, even if it's on a different topic than the "
        "conversation above — return the question completely "
        "UNCHANGED. Do not add topic context to a question that "
        "doesn't need it.\n"
        "Respond with ONLY the question (rewritten or unchanged), "
        "nothing else."
    )
    try:
        raw = await _backend_completion(prompt, max_tokens=60, timeout=4.0)
        if not raw or not raw.strip():
            return question
        return raw.strip()
    except Exception:
        return question