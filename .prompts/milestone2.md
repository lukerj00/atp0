# Milestone 2 — Orchestrator V0 (Direct Proving Loop + Model Gateway Stub)

## Goal
Build the first usable end-to-end **ATP loop** that:
- accepts a **formal Lean 4 theorem statement** (plus optional imports/decls),
- calls an **LLM-agnostic Model Gateway** to generate proof candidates,
- calls **LeanWorker** to verify candidates,
- returns the first verified proof (if found) plus structured logs.

**No training.** No lemma decomposition yet (that’s Milestone 4).  
This milestone establishes the **Prove → Refine** core loop and the model abstraction.

---

## Dependencies
Milestone 1 must be complete and running:
- LeanWorker `POST /check` endpoint
- Deterministic Lean template and stable diagnostics schema

---

## Scope (what to build)

### Deliverables
1. **Orchestrator HTTP service** (FastAPI recommended)
2. **Model Gateway interface** (LLM-agnostic) with:
   - a **Dummy backend** (for deterministic testing)
   - a **Real backend adapter stub** (OpenAI/Gemini/DeepSeek placeholder)
3. **Direct proving loop**:
   - generate N candidates
   - verify all with LeanWorker
   - if fail, run repair loop for top-K failures
4. **Failure parsing + prioritization** using LeanWorker diagnostics
5. **Simple local cache**:
   - avoid re-checking identical candidate proofs for identical goals
6. **Artifacts/logging**:
   - return final Lean proof + attempt history summary

---

## Non-goals (Milestone 2)
- No interactive stepwise goal-state REPL (still batch verification via LeanWorker)
- No lemma planning / DAGs / blueprint search
- No retrieval over mathlib (Milestone 3)
- No GCP deployment required (local docker-compose is enough)

---

## Service Layout (recommended)

orchestrator/
Dockerfile
requirements.txt
app/
init.py
server.py
config.py
schemas.py
model_gateway/
init.py
base.py
dummy.py
openai_stub.py
prove_loop/
init.py
runner.py
scoring.py
cache.py
leanworker_client.py
prompts/
prover_prompt.txt
repair_prompt.txt
benchmarks/
cases.jsonl
run_tests.py

yaml
Copy code

---

## API Spec

### `GET /healthz`
```json
{ "ok": true }
POST /prove
Attempt to prove a theorem.

Request schema
json
Copy code
{
  "job_id": "optional-string",
  "imports": ["Mathlib"],
  "extra_prelude": "optional-string",
  "decls": "optional-string",
  "theorem_name": "T",
  "theorem_statement": "∀ n : Nat, n = n",
  "budget": {
    "max_rounds": 4,
    "candidates_per_round": 12,
    "repairs_per_round": 6,
    "timeout_ms_per_check": 15000,
    "max_total_checks": 60
  },
  "model": {
    "backend": "dummy|openai|gemini|deepseek|local",
    "model_name": "optional",
    "temperature": 0.7
  }
}
Response schema
json
Copy code
{
  "job_id": "string",
  "ok": true,
  "final_proof": {
    "proof_block": "by\n  intro n\n  rfl",
    "lean_file": "string (full file as checked)",
    "theorem_name": "T"
  },
  "stats": {
    "rounds_used": 2,
    "checks_used": 17,
    "time_ms_total": 45210,
    "cache_hits": 8
  },
  "attempts": [
    {
      "round": 1,
      "candidate_id": "r1_c3",
      "proof_block": "by\n  ...",
      "lean_ok": false,
      "error_class": "tactic_failed",
      "message_excerpt": "string",
      "score": 0.62
    }
  ]
}
Semantics
ok=true iff a proof candidate verifies with LeanWorker.

Orchestrator returns the first proven candidate found under the search policy.

attempts may be truncated to keep responses small (store full logs to disk later).

Model Gateway (LLM-agnostic)
Core interface
Implement a provider-independent interface:

python
Copy code
class ModelGateway(Protocol):
    def propose(self, problem: ProverInput, n: int) -> list[str]:
        """Return n proof candidate tactic scripts (each must be a Lean `by ...` block)."""

    def repair(self, problem: ProverInput, failed_proof: str, diag: LeanDiag, n: int) -> list[str]:
        """Return n repaired candidate proofs given Lean error feedback."""
Candidate output constraints (important)
Each candidate must be a Lean proof block:

begins with by

short (recommend <= 40 lines in V0)

No commentary in output. If the backend produces extra text, gateway must strip it.

Required backends
Dummy backend

Deterministically returns a small set of candidates (e.g., by simp, by intro; rfl, by aesop)

Used to validate integration and regression tests without paid API calls.

Stub backend(s)

Provide placeholders for real providers:

OpenAI adapter stub with “call_model(prompt)”

Gemini stub

DeepSeek stub

Actual provider integration may be implemented later; V0 can keep stubs.

Prove → Refine Loop (Core algorithm)
Inputs
theorem statement (+ imports/decls)

LeanWorker endpoint

ModelGateway backend

budget parameters

Outputs
success proof block or failure report

Algorithm (concrete)
Pseudo-code:

Normalize problem (trim whitespace, canonicalize statement, sanitize theorem name).

Initialize:

seen_candidates set

attempt_log list

checks_used = 0

For round in 1..max_rounds:

Generate candidates:

cand_list = gateway.propose(problem, candidates_per_round)

Deduplicate + filter:

remove any already in seen_candidates

if empty, continue

Verify candidates (parallelize if possible):
For each candidate in cand_list:

if checks_used >= max_total_checks: stop

call LeanWorker /check with {imports, extra_prelude, decls, theorem_name, stmt, proof=candidate}

record attempt with:

ok/fail

error_class

message excerpt

if ok: return success immediately

If none succeed:

Score failures (see scoring section).

Select top repairs_per_round failures.

For each selected failure:

repairs = gateway.repair(problem, failed_candidate, diag, n=1 or small)

verify repairs immediately (same dedupe + cache checks)

if any ok: return success

Return failure response with best attempts.

Parallelization (optional but recommended)
Use asyncio + httpx to call LeanWorker concurrently (bounded semaphore).

This is a big throughput win.

Failure scoring and prioritization
Implement lightweight scoring that decides which failures are “promising to repair”.

Required error_class buckets
Use LeanWorker’s error_class, fallback to parsing message text.

Heuristic priority (best to repair first):

unsolved_goals (close; suggests tactic changes)

tactic_failed (often fixable with different lemma/tactic)

type_mismatch (may be fixable but sometimes indicates wrong approach)

unknown_identifier (often import/retrieval-related; mostly Milestone 3)

parse_error (easy fix but indicates the model is sloppy)

other

Score formula example:

base score = by class priority

small bonus if message is short/simple (less confusion)

bonus if candidate contains known good tactics (simp, aesop, linarith, ring)

Cache (required minimal)
Implement a local cache to avoid re-checking the same candidate repeatedly.

Key
env_hash: hash(imports + extra_prelude + decls)

stmt_hash: hash(theorem_statement normalized)

cand_hash: hash(candidate proof normalized)

Value
leanworker response summary: ok/fail, error_class, message excerpt, time_ms

Store:

in-memory dict first

optional SQLite later

Prompts (stored as files, versioned)
Even if you don’t integrate a real LLM yet, implement prompt templates because they are part of the stable interface.

Prover prompt (propose)
Contents should instruct:

output only Lean code

proof must start with by

no imports, no theorem header (just the proof block)

Repair prompt (repair)
Contents should include:

the goal statement

the failed proof block

Lean error excerpt

instruct to return a corrected proof block only

The agent should create minimal default prompts in app/prove_loop/prompts/.

Acceptance tests (Milestone 2)
Setup
Start LeanWorker (Milestone 1)

Start Orchestrator

Tests
Trivial theorem: should succeed using dummy backend (e.g., by intro; rfl)

Simp theorem: should succeed using by simp from dummy backend

Harder theorem: expected fail, but returns structured attempts with error classes

Provide benchmarks/run_tests.py that calls /prove and asserts:

response schema contains ok, attempts, stats

at least one success case returns final_proof.lean_file

Definition of Done
Orchestrator runs locally with config pointing at LeanWorker

/prove works end-to-end with dummy backend on at least 2 success cases

Repair loop runs at least once on a failure case (even if it still fails)

Cache prevents re-checking identical candidates within a job

Logs/attempt history returned in response

Notes for later milestones
Milestone 3 will add retrieval (fixes unknown identifiers / library alignment)

Milestone 4 will add planning (lemma decomposition, blueprint edits)

Later will replace batch check with interactive Lean goal-state extraction (big win)

makefile
Copy code
::contentReference[oaicite:0]{index=0}