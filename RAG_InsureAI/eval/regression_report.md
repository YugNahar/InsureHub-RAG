# Regression Report — 2026-07-06 11:36:19 UTC

- **Host**: `http://localhost:8002`
- **Total**: 39
- **Passed**: 12
- **Failed**: 14
- **Skipped**: 13

---

## Per-Category Summary

| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| calculation | 1 | 1 | 0 | 0 |
| compound | 2 | 1 | 1 | 0 |
| enumeration | 3 | 3 | 0 | 0 |
| follow_up | 4 | 0 | 0 | 4 |
| greeting | 3 | 0 | 3 | 0 |
| modifier | 3 | 0 | 0 | 3 |
| off_topic | 3 | 0 | 3 | 0 |
| ordinal | 3 | 0 | 0 | 3 |
| quoted_comparison | 2 | 0 | 2 | 0 |
| short_followup | 2 | 0 | 0 | 2 |
| standalone | 5 | 5 | 0 | 0 |
| topic_switch | 1 | 0 | 0 | 1 |
| typo_correction | 2 | 2 | 0 | 0 |
| unsupported | 4 | 0 | 4 | 0 |
| user_statement | 1 | 0 | 1 | 0 |

## Failed Cases

### unsupported_01 (unsupported)

- **Query**: `what is the GST rate on insurance premiums?`
- **Answer**: General knowledge (not from uploaded documents): The GST (Goods and Services Tax) rate on insurance premiums varies by country. In some countries, such as India, the GST rate on insurance premiums is 18%. However, in other countries, such as the United Arab Emirates (where Dubai is located), VAT (Va
- **Expect `should_refuse`**: expected=True  detail=refusal=False

### unsupported_02 (unsupported)

- **Query**: `what is the current repo rate set by RBI?`
- **Answer**: General knowledge (not from uploaded documents): The current repo rate set by the Reserve Bank of India (RBI) is not specified in the provided context, as the context appears to be related to health insurance plans and does not mention the RBI or repo rates. The RBI's repo rate is the rate at which 
- **Expect `should_refuse`**: expected=True  detail=refusal=False

### unsupported_03 (unsupported)

- **Query**: `who is the CEO of LIC right now?`
- **Answer**: General knowledge (not from uploaded documents): The CEO of LIC (Life Insurance Corporation of India) is not specified in the provided context, and the documents appear to be related to health insurance plans offered by other companies. To find the current CEO of LIC, I would recommend checking the 
- **Expect `should_refuse`**: expected=True  detail=refusal=False

### unsupported_04 (unsupported)

- **Query**: `what is IDV in motor insurance?`
- **Answer**: General knowledge (not from uploaded documents): IDV in motor insurance stands for Insured Declared Value. It is the maximum amount that an insurance company will pay in the event of a claim, typically in cases of total loss or theft of the vehicle. The IDV is usually calculated based on the manufac
- **Expect `should_refuse`**: expected=True  detail=refusal=False

### greeting_01 (greeting)

- **Query**: `hi`
- **Answer**: Hello. How can I assist you with your insurance-related query today?
- **Expect `no_retrieval_expected`**: expected=True  detail=sources_empty=False  sources=2

### greeting_02 (greeting)

- **Query**: `hello there`
- **Answer**: Hello. How can I assist you with your insurance-related questions today?
- **Expect `no_retrieval_expected`**: expected=True  detail=sources_empty=False  sources=2

### greeting_03 (greeting)

- **Query**: `good morning`
- **Answer**: Good morning. Is there something I can help you with regarding your insurance documents or would you like to ask a specific question?
- **Expect `no_retrieval_expected`**: expected=True  detail=sources_empty=False  sources=2

### compound_01 (compound)

- **Query**: `what is a deductible and what is the GST rate on insurance premiums`
- **Answer**: General knowledge (not from uploaded documents):   A deductible is a specific amount that the insured must pay out of pocket for a medical expense before the insurance coverage kicks in. It is a common feature in many health insurance plans, where the insured is required to pay a certain amount, and
- **Expect `should_refuse`**: expected=True  detail=refusal=False

### quoted_comparison_01 (quoted_comparison)

- **Query**: `difference between "floater policy" and "specific policy" in fire insurance`
- **Answer**: [LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kwhwdzdefpmvn6zmhd2bsk89` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99833, Requested 1928. Please try again in 25m21.504s. Need more tokens? U
- **Expect `should_refuse_if_either_term_absent`**: expected=True  detail=refusal=False

### quoted_comparison_02 (quoted_comparison)

- **Query**: `what's the difference between "reinsurance" and "coinsurance"`
- **Answer**: [LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kwhwdzdefpmvn6zmhd2bsk89` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99821, Requested 1657. Please try again in 21m16.992s. Need more tokens? U
- **Expect `should_refuse_if_either_term_absent`**: expected=True  detail=refusal=False

### off_topic_01 (off_topic)

- **Query**: `what's the weather today`
- **Answer**: [LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kwhwdzdefpmvn6zmhd2bsk89` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99691, Requested 1861. Please try again in 22m20.928s. Need more tokens? U
- **Expect `is_off_topic_refusal`**: expected=True  detail=refusal=False

### off_topic_02 (off_topic)

- **Query**: `who won the cricket world cup`
- **Answer**: [LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kwhwdzdefpmvn6zmhd2bsk89` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99650, Requested 1914. Please try again in 22m31.296s. Need more tokens? U
- **Expect `is_off_topic_refusal`**: expected=True  detail=refusal=False

### off_topic_03 (off_topic)

- **Query**: `can you write me a python script`
- **Answer**: [LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kwhwdzdefpmvn6zmhd2bsk89` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99630, Requested 1755. Please try again in 19m56.639999999s. Need more tok
- **Expect `is_off_topic_refusal`**: expected=True  detail=refusal=False

### user_statement_01 (user_statement)

- **Query**: `I recently bought a term life insurance plan`
- **Answer**: [LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01kwhwdzdefpmvn6zmhd2bsk89` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99585, Requested 1775. Please try again in 19m35.04s. Need more tokens? Up
- **Expect `no_retrieval_expected`**: expected=True  detail=sources_empty=False  sources=1

## Skipped Cases

- **follow_up_01** (follow_up): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **follow_up_02** (follow_up): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **follow_up_03** (follow_up): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **follow_up_04** (follow_up): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **ordinal_01** (ordinal): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **ordinal_02** (ordinal): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **ordinal_03** (ordinal): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **modifier_simple_01** (modifier): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **modifier_detail_01** (modifier): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **modifier_example_01** (modifier): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **short_followup_01** (short_followup): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **short_followup_02** (short_followup): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query
- **fresh_topic_after_followup_01** (topic_switch): EvalRequest has no 'history' field; history-bearing cases cannot be submitted via /eval/query

