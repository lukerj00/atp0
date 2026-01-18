# Staircase V2: Multi-Round Verification-Guided Repair + Caching

## Overview

V1 adds strategy-diverse pass@K + parallel verification + a single repair round.

V2 upgrades to a **multi-round verification-guided repair loop**:
1. **Multi-round propose → verify → repair** (bounded, deterministic budgets)
2. **Per-problem caching** (avoid re-verifying duplicate proofs)
3. **Error-class-aware repair policies** (different prompts based on error type)

Still:
- Gemini-only
- Kimina server for verification (default, 50-100x faster than local)
- Single file structure (`benchmark_v2.py`)
- No retrieval (deferred to V3)
- No blueprint planning / lemma decomposition
- No stepwise tactic sessions / goal-state querying

This stage takes you from "K samples + one repair" to a robust **iterative prover**.

---

## Architecture

```
┌──────────────────────┐
│   Problem Set        │
└───────┬──────────────┘
        ▼
┌─────────────────────────────────────────────┐
│ Round 0: Propose K diverse candidates       │
└───────┬─────────────────────────────────────┘
        ▼
┌─────────────────────────────────────────────┐
│ Verify batch (server) + cache results       │
└───────┬─────────────────────────────────────┘
        │ success? ───────────────▶ SUCCESS
        │
        ▼ (fail)
┌─────────────────────────────────────────────┐
│ Round 1..R: Repair loop                     │
│ - select best failures (by error class)     │
│ - generate M repairs per failure            │
│ - verify (dedupe via cache)                 │
│ - repeat until budget exhausted             │
└───────┬─────────────────────────────────────┘
        ▼
   SUCCESS / FAIL
```

---

## What V2 Adds (relative to V1)

### A) Multi-round repair loop (bounded)
Instead of one repair round, do up to `max_rounds` (default 3), with strict budgets.

### B) Per-problem proof cache
Avoid re-verifying:
- Identical proof blocks (by normalized hash)
- Track seen proofs to dedupe before sending to server

### C) Error-class-specific repair policies
Classify errors and adjust repair prompts:
- `unknown_identifier` → suggest using simp/exact with common lemma patterns
- `type_mismatch` → suggest explicit types/casts
- `tactic_failed` / `unsolved_goals` → try different tactic families
- `parse_error` → formatting/indent fixes

---

## Implementation (single file: benchmark_v2.py)

### Key functions to add/modify:

#### 1) `classify_error(error_excerpt: str) -> str`
Returns: `unknown_identifier`, `type_mismatch`, `tactic_failed`, `unsolved_goals`, `parse_error`, or `other`

#### 2) `normalize_proof(proof: str) -> str`
Normalize whitespace for deduplication.

#### 3) `generate_repairs_v2(theorem, failed_proof, error_excerpt, error_class, r) -> list[str]`
Error-class-aware repair prompt.

#### 4) `process_problem_v2(prob, args, executor) -> dict`
Multi-round loop:
```
seen_proofs = set()
all_attempts = []

# Round 0: propose
proofs = generate_proofs(theorem, k)
proofs = dedupe(proofs, seen_proofs)
results = verify_batch(proofs)
all_attempts.extend(results)
if any success: return winner

# Rounds 1..max_rounds: repair
for round in range(1, max_rounds + 1):
    if total_verifications >= budget: break

    # Select best failures (diverse by error class)
    failures = select_best_failures(all_attempts, top_n=args.top_failures)

    for failure in failures:
        error_class = classify_error(failure.error_excerpt)
        repairs = generate_repairs_v2(theorem, failure.proof, failure.error_excerpt, error_class, args.repairs_per_failure)
        repairs = dedupe(repairs, seen_proofs)

        if repairs:
            results = verify_batch(repairs)
            all_attempts.extend(results)
            if any success: return winner

return failure with all_attempts
```

---

## CLI Arguments (V2)

```
--k                     Initial candidates (default: 8)
--max-rounds            Max repair rounds (default: 3)
--top-failures          Failures to repair per round (default: 2)
--repairs-per-failure   Repairs per failure (default: 3)
--max-verifications     Total verification budget (default: 50)
--timeout               Server timeout per batch (default: 60)
--problems              Problem file (default: problems.json)
--local                 Use local Lean instead of server
```

---

## Prompts

### Propose prompt
Same as V1 (JSON K candidates with strategy tags).

### Repair prompt (error-class-aware)
```
You are repairing a Lean 4 proof.

THEOREM:
{theorem}

FAILED PROOF:
{failed_proof}

LEAN ERROR:
{error_excerpt}

ERROR TYPE: {error_class}

TASK: Produce exactly {r} DIFFERENT repaired proof candidates.

RULES:
- Output ONLY valid JSON: {{"repairs":[{{"strategy":"<tag>","proof":"<proof starting with by>"}}, ...]}}
- Each proof MUST start with `by`
- Do NOT use `sorry`
- Try different approaches based on error type:
  - If unknown_identifier: check lemma names, try simp/exact with variants
  - If type_mismatch: add explicit type annotations or casts
  - If tactic_failed/unsolved_goals: try a completely different tactic family
  - If parse_error: fix syntax/indentation

STRATEGY TAGS: simp, aesop, induction, cases, algebra, have_chain, calc, inequalities

R = {r}
```

---

## Results / Logging

Per-problem result includes:
- `rounds_used`: how many rounds before success/failure
- `verifications_used`: total proofs verified
- `winning_round`: which round found the solution (0 = propose, 1+ = repair)
- `all_attempts`: full list of verification results

Summary includes:
- `pass@1`, `pass@K`, `final_score`
- `mean_verifications_per_problem`
- `mean_rounds_per_solved`
- `error_class_histogram`

---

## Success Criteria

1. Multi-round loop runs correctly under budgets
2. Cache prevents duplicate verifications
3. Error classification works and affects prompts
4. Improved score vs V1 on same problems
5. Full trace artifacts for debugging
