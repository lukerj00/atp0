# Staircase V1: pass@K + Strategy Diversity + Parallel Verification (Gemini-only) + 1-Step Repair

## Overview

V0 establishes a baseline: **one Gemini call → compile once → score**.

V1 upgrades the baseline while staying simple and debuggable by adding:
1. **Strategy-diverse multi-sampling (pass@K)**: generate *K* proof candidates that intentionally try *different* approaches
2. **Parallel verification**: compile candidates concurrently (bounded)
3. **Verification-guided selection**: accept the first proof that Lean verifies
4. **Minimal repair loop**: if all K fail, do **one** additional Gemini call conditioned on the *best* Lean error and request **R repaired variants** (R small, e.g. 4), verified in parallel

**Still no retrieval, no planning/lemmas, no caching across problems.**  
But this is the first real “propose → verify → refine” baseline.

---

## Why V1 (what it improves)
- pass@K is the simplest way to get big gains.
- forcing **diverse strategies** reduces “K copies of the same failing idea”.
- parallel compile makes the extra candidates cheap in wall-clock time.
- a single repair round leverages Lean feedback without complex orchestration.

---

## Architecture

┌──────────────────┐
│ PutnamBench (20) │
└───────┬──────────┘
▼
┌───────────────────────────────┐
│ Gemini Flash: propose K │ (strategy-diverse)
└───────┬───────────────────────┘
▼
┌───────────────────────────────┐
│ Lean Verifier (parallel) │ compile K candidates concurrently
└───────┬───────────────────────┘
│ pass? ────────────────▶ SUCCESS
│
▼ (all fail)
┌───────────────────────────────┐
│ Gemini Flash: repair R │ conditioned on best error
└───────┬───────────────────────┘
▼
┌───────────────────────────────┐
│ Lean Verifier (parallel) │ compile R repairs concurrently
└───────┬───────────────────────┘
▼
SUCCESS / FAIL

yaml
Copy code

---

## Components to Upgrade / Add

### 1) Problem Loader (`problems/`)
Same as V0.

### 2) Gemini Client (`src/llm.py`) — add strategy-diverse K outputs + repair variants
V0: `generate_proof(statement) -> str`

V1 requires:

#### `generate_proofs_diverse(statement: str, k: int, temperature: float) -> list[str]`
- Returns **K** proof blocks, each starting with `by`
- Must be *explicitly strategy-diverse* (see prompting rules below)
- Implementation options:
  - **Preferred**: one call that asks for JSON with K entries, each labeled with a strategy tag
  - **Fallback**: K separate calls, each with a different requested strategy tag (best diversity)

#### `repair_proofs_diverse(statement: str, failed_attempts: list[dict], best_error: str, r: int) -> list[str]`
- Returns **R** repaired proof variants in one call (or R calls) with explicit diversity
- Condition on:
  - theorem statement
  - the single best failed proof (or top 2)
  - Lean error excerpt
- Output: R proof blocks starting with `by`

---

### 3) Lean Verifier (`src/verifier.py`) — structured diagnostics (same as earlier V1)
Return:
```python
VerifyResult = {
  "success": bool,
  "time_ms": int,
  "stdout": str,
  "stderr": str,
  "error_excerpt": str | None,
  "error_class": str | None,
}
Heuristic error classes:

unknown_identifier, type_mismatch, tactic_failed, unsolved_goals, parse_error, other

Extract error_excerpt = first ~20 lines of stderr or first error block.

4) Parallel Candidate Verification (src/verify_pool.py) — NEW
Implement a bounded-parallel verifier runner.

Requirements:

Use asyncio + subprocess / thread pool (implementation choice)

Cap concurrency:

--max-parallel default 4 (or 8 on strong machines)

Verify candidates in parallel and return the first success as soon as it completes (“race to first proof”).

Still record results for all candidates that completed.

API:

python
Copy code
verify_many(candidates: list[str], timeout_ms: int, max_parallel: int) -> list[VerifyResult]
first_success(results) -> (idx, result) | None
5) Candidate Evaluator (src/eval.py) — best failure selection
Pick best failure for repair.

Best failure heuristic:
Priority order:

unsolved_goals

tactic_failed

type_mismatch

unknown_identifier

parse_error

other

Tie-breakers:

shorter stderr

faster compile time

Return:

best_failed_proof

best_error_excerpt

best_error_class

plus the full attempt list (for logging / for repair prompt context)

6) Benchmark Runner (src/benchmark.py) — pass@K + parallel verification + repair-R
Per problem:

Propose phase

call generate_proofs_diverse(stmt, k=K)

verify all K in parallel (verify_many)

if any success: accept the first proven proof

Repair phase (one round only)

if all K fail:

select best failure (eval.py)

call repair_proofs_diverse(..., r=R) (default R=4)

verify all R in parallel

if any success: accept the first proven repair

Log everything.

Prompting (V1) — Ensure K Different Strategies
Core principle
Gemini must be instructed to produce K materially different proof strategies, not K minor variants.

Strategy palette (fixed list)
Each candidate must choose a different primary strategy from this set (reuse only if K > list length):

simp / rewriting (simp, simpa, simp [..])

aesop / automation (aesop, aesop?, simp?)

induction (structural induction with explicit cases)

cases / by_cases / Classical split

algebraic normalization (ring, nlinarith, linarith, field_simp)

have chain (derive intermediate facts, then exact)

calc proof with rewrite steps

order/inequality tactics (nlinarith, linarith, omega, positivity)

Propose prompt (single-call JSON format preferred)
Use a single prompt that requests JSON with K entries:

perl
Copy code
You are a Lean 4 theorem prover.

TASK:
Given the theorem statement below, produce exactly K DIFFERENT Lean proof candidates.

HARD RULES:
- Output ONLY valid JSON (no markdown, no commentary).
- JSON schema:
  {"candidates":[{"strategy":"<tag>","proof":"<Lean proof block starting with by>"}, ...]}
- Each proof MUST start with `by`.
- Do NOT include imports, theorem headers, or explanations.
- Do NOT use `sorry`.
- Candidates must be DIVERSE: use different primary strategies and substantially different proof structures.
- Use Mathlib tactics: simp, simpa, aesop, ring, nlinarith, linarith, omega, norm_num, field_simp, polyrith.
- If you use Classical reasoning, add `classical` inside the proof block.

STRATEGY TAGS (choose distinct tags when possible):
simp, aesop, induction, cases, algebra, have_chain, calc, inequalities

THEOREM STATEMENT:
<stmt>

K = <K>
Client-side requirements:

Parse JSON strictly

If a candidate’s proof doesn’t start with by, drop it or coerce (prefer drop + request regen if too many invalid)

If many candidates repeat the same strategy, optionally do a second call asking explicitly for missing tags (but keep V1 minimal: one call is fine)

Repair prompt (R variants, diverse)
powershell
Copy code
You are repairing Lean 4 proofs.

THEOREM STATEMENT:
<stmt>

BEST FAILED PROOF:
<failed_proof>

LEAN ERROR (excerpt):
<error_excerpt>

TASK:
Produce exactly R DIFFERENT repaired proof candidates.

HARD RULES:
- Output ONLY valid JSON:
  {"repairs":[{"strategy":"<tag>","proof":"<Lean proof block starting with by>"}, ...]}
- Each proof MUST start with `by`.
- Do NOT include imports, theorem headers, or explanations.
- Do NOT use `sorry`.
- Repairs must be diverse (different strategy tags if possible).
- Prefer minimal changes when using the same strategy; otherwise attempt a different approach.

STRATEGY TAGS:
simp, aesop, induction, cases, algebra, have_chain, calc, inequalities

R = <R>
Results / Logging (upgrade)
Per problem save:

statement.txt

propose_candidates.json (raw model JSON)

attempts.jsonl (one record per candidate verified)

if repair used:

repair_candidates.json

repair_attempts.jsonl

final.lean (winning proof file if success)

summary.json (per problem)

Overall save:

results/summary.json with:

score X/20

pass@1 (first candidate only)

pass@K

repair_success_count

total runtime stats (mean verify time, etc.)

File Structure (from V0)
New:

src/verify_pool.py (parallel verifier)

src/eval.py (best failure selection; can be small)

Updated:

src/llm.py (diverse K outputs + diverse repairs)

src/verifier.py (structured diagnostics)

src/benchmark.py (pass@K + parallel verify + repair-R)

CLI Entrypoint
Update run_benchmark.py args:

--k (default 8)

--r (default 4)

--temperature (default 0.7)

--timeout-ms (default 15000)

--max-parallel (default 4)

--repair (on/off, default on)

Example:

bash
Copy code
python run_benchmark.py --k 8 --r 4 --max-parallel 4 --temperature 0.7 --timeout-ms 15000 --repair on
Success Criteria
Runnable: python run_benchmark.py executes end-to-end

Parallel: candidates compile concurrently (bounded), and runner returns promptly on first success

Diversity: logs show multiple distinct strategy tags per problem (when K allows)

Scored: outputs X/20 plus pass@1, pass@K, and repair success rate

Traceable: all candidates + stderr excerpts are stored for inspection

Explicit Non-Goals (still deferred)
No retrieval / mathlib search

No caching across problems

No lemma planning / blueprint search

No stepwise goal-state guidance from Lean (this arrives in later staircase stages)

No multi-model routing

These arrive in V2+.

makefile
Copy code
::contentReference[oaicite:0]{index=0}