# Milestone 5 — Stepwise Tactic Mode (REPL-like) + Parallel Verification + Budgeting

## Goal
Replace (or augment) batch “whole proof block” attempts with a **stepwise tactic mode** that:
- applies **one tactic at a time**,
- queries Lean for updated goals after each step,
- uses the resulting proof state to guide the next tactic.

This is the biggest practical jump toward Aleph-like efficiency:
- fewer wasted long generations,
- much better repair behavior,
- better search control (branching at each step),
- more informative feedback than “compile / fail”.

In the same milestone, add:
- **parallel candidate evaluation** (bounded concurrency)
- **explicit budget manager** (tokens, Lean checks, time, steps)
- improved **attempt tracing** for later analysis

**No training.** Still LLM-agnostic.

---

## Dependencies
- Milestone 1: LeanWorker
- Milestone 2: Orchestrator proving loop
- Milestone 3: Retriever + RAG
- Milestone 4: Blueprint planning + lemma decomposition

---

## Scope (what to build)

### Deliverables
1. Extend LeanWorker to support a **stepwise checking API**:
   - initialize a proof session
   - apply a tactic
   - read the current goal state
   - finalize / check completion
2. Implement **stepwise proving** in Orchestrator:
   - for lemma goals and main theorem
   - integrated with blueprints
3. Add **parallelism**:
   - run multiple sessions/candidates concurrently
4. Add **BudgetManager**:
   - per theorem job: max wall time, max Lean calls, max steps, max model calls
5. Add **structured traces**:
   - store full tactic sequences with intermediate goal states
   - store failure points for repairs

---

## Non-goals (Milestone 5)
- No full Lean language server integration in editor (this is backend only)
- No distributed cluster scheduler (single-machine concurrency is enough)
- No learned tactic policies / value models

---

## LeanWorker: Stepwise API

### Why this is needed
Batch compilation provides only coarse feedback. Stepwise mode allows:
- goal-state aware generation
- targeted repairs at the failing step
- measurable progress (goals reduced, simp closed goals, etc.)

### Implementation approach (recommended)
Use a **persistent Lean process** per session and communicate through a structured interface.

There are multiple ways; pick one that is practical in V0:

#### Option A (recommended for V0.5): “Lean REPL script runner”
- Maintain a session by incrementally extending a Lean file:
  - append `by` block tactics line-by-line,
  - after each append, run `lake env lean` and parse the **goal state** from output.
- This is slower than a true REPL but easiest to implement and still yields stepwise behavior.

#### Option B (preferred long-term): Lean server / interactive protocol
- Use Lean’s server APIs to:
  - keep an environment loaded
  - send edits
  - retrieve diagnostics and goal states efficiently

**Milestone 5 should implement Option A** if Option B is too complex. The Orchestrator interface should be designed so LeanWorker can later swap to Option B without breaking clients.

---

## Stepwise API Endpoints (LeanWorker)

### `POST /session/start`
Create a new proof session.

#### Request
```json
{
  "imports": ["Mathlib"],
  "extra_prelude": "",
  "decls": "",
  "theorem_name": "T",
  "theorem_statement": "...",
  "mode": "tactic",
  "options": { "timeout_ms": 15000 }
}
Response
json
Copy code
{
  "session_id": "uuid",
  "ok": true,
  "initial_goal": {
    "pretty": "⊢ ...",
    "raw": "string"
  }
}
POST /session/step
Apply one tactic step (or a short block) to the session.

Request
json
Copy code
{
  "session_id": "uuid",
  "tactic": "intro n",
  "options": { "timeout_ms": 15000 }
}
Response
json
Copy code
{
  "session_id": "uuid",
  "ok": true,
  "done": false,
  "goal_state": {
    "pretty": "n : Nat\n⊢ n = n",
    "raw": "string"
  },
  "diagnostics": [
    { "severity": "error", "message": "...", "error_class": "tactic_failed", "raw": "..." }
  ],
  "time_ms": 1200
}
POST /session/close
Terminate session, delete temp files, free resources.

Request
json
Copy code
{ "session_id": "uuid" }
Response
json
Copy code
{ "ok": true }
GET /session/{id}
(Optional) fetch current stored script and goal state (debugging).

LeanWorker: Goal state extraction (Option A)
File strategy
LeanWorker internally maintains:

Main.lean with the theorem and an evolving by block:

theorem T : stmt := by

then tactic lines appended

After each step:

run lake env lean Main.lean

parse output for:

errors

remaining goals

Getting goal states from Lean output
You will need a stable way to print goal states.

Practical V0 approach:

Insert show or set_option pp.all true and use tactics that report goals on failure.

Better: use #check? Not relevant.

Alternative: use a custom Lean tactic that prints goals at a marker.

Recommended approach for Milestone 5 (most reliable):

Wrap each step with a helper macro that forces Lean to print goals using tactic tracing.

Use set_option trace.Tactic.goal true or similar tactic traces (Lean has tracing facilities; exact trace key may vary).

Then parse the trace output to extract goals.

If tracing keys are unstable, fallback to “detect completion only” (done when Lean succeeds with no errors) and treat goal_state as “unknown”. But the milestone’s aim is to get some goal text.

Orchestrator: Stepwise proving
Replace proof generation calls
Instead of:

propose full by ... scripts

Use:

propose next tactic given current goal_state

Model Gateway interface updates
Add:

python
Copy code
def propose_next_tactic(goal_state: str, context: PromptContext, n: int) -> list[str]
def repair_next_tactic(goal_state: str, last_tactic: str, diag: LeanDiag, context: PromptContext, n: int) -> list[str]
Stepwise prove loop (per goal)
Parameters:

max_steps_per_goal (e.g., 80)

branch_width (e.g., 8 candidate tactics per step)

max_sessions_parallel (e.g., 8)

Algorithm (beam over sessions):

Start K sessions for the same goal (or 1 session and branch later).

For step in 1..max_steps:

For each active session:

get current goal_state

ask model for n tactic candidates

apply each tactic to a cloned session OR sequentially with rollback

Keep top K sessions according to progress heuristic (below).

Stop when any session reports done=true.

If all sessions dead (errors), trigger:

repair prompt

or go back to Plan/blueprint refinement.

Session cloning strategies (important choices)
LeanWorker Option A cannot “clone” a running Lean process, but it can:

duplicate the session directory (copy Main.lean file) to fork states.

Two choices:

C1: Copy-on-branch: each branch lives in its own temp dir; branching copies the file.

C2: Rebuild-from-log: store tactics history and reconstruct state by replaying from scratch (slower).

Choose C1 for V0.

Progress heuristics (stepwise beam scoring)
Score each active session to decide which branches to keep:

Completed proof: highest priority

Fewer remaining goals: better

Shorter goal text (proxy): better

Error-free recently: better

Avoid loops: penalize repeating the same goal state hash

You can approximate “number of goals” by parsing printed goal state delimiters.

Parallelization
What to parallelize
Candidate tactic applications across sessions

LeanWorker /session/step calls

(Optional) model calls for different sessions

Implementation
Orchestrator uses asyncio + bounded semaphore.

LeanWorker may also run multiple sessions concurrently; enforce limits.

BudgetManager (required)
Budgets to track per theorem job
max_wall_time_ms (overall)

max_model_calls

max_lean_calls

max_sessions_created

max_steps_total

max_checks_total (if still using batch checks)

Policy
Stop when budget exhausted; return best partial attempt log.

Adaptive policy:

start small branch width

increase only if stuck (optional)

Tracing and artifacts
Required trace outputs
For each goal (lemma or main theorem):

blueprint id/version

session ids explored

tactic sequence for each branch (until failure/success)

key goal_state snapshots (at least when branching or failing)

Lean diagnostics at failure points

total costs/time

Store traces:

in JSONL for easy analysis

optionally upload to artifact store later

Acceptance tests (Milestone 5)
Tests must demonstrate stepwise behavior
A proof solvable in 2–5 tactics (e.g., ∀ n, n = n)

show session steps and goal_state changes

A proof where the first tactic fails, repair suggests alternative tactic

A case where beam branching helps (two tactic options; only one works)

Assertions:

response includes tactic sequence (not only final proof)

LeanWorker sessions are created and stepped

budgets respected (caps enforced)

Definition of Done
LeanWorker exposes /session/start, /session/step, /session/close

Orchestrator can solve at least simple theorems via stepwise tactics

Beam search over sessions is implemented with a progress heuristic

Parallel calls are bounded and stable under load

BudgetManager stops runaway searches deterministically

Full traces are produced for success and failure cases

Notes for Milestone 6+
Replace Option A session mode with true Lean interactive protocol for speed

Add blueprint beam search (multiple plans in parallel)

Add import minimization + better premise selection

Add more robust goal-state parsing with Lean metaprogramming hooks

makefile
Copy code
::contentReference[oaicite:0]{index=0}