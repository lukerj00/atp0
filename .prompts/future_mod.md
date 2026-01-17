```md
# Optional Improvements / Modifications to Try (Post-Milestone Enhancements)

This document lists targeted modifications you can try to improve solve rate, cost, and stability beyond Milestones 1–6. These are ordered roughly by **impact-to-effort** and aligned with what “Aleph-like” systems likely do in practice (verification-guided, blueprint-driven, cost-aware).

Each item includes:
- **What to change**
- **Why it helps**
- **How to implement (high-level)**
- **Risks / gotchas**

---

## 1) Upgrade LeanWorker from batch checks to true interactive goal-state queries (HIGH impact)
### What
Replace “write file + `lake env lean`” with a **persistent Lean server / interactive protocol** (incremental elaboration and goal inspection).

### Why
- 10–100× faster feedback in tight loops
- better goal-state extraction and stateful sessions
- enables fine-grained progress heuristics

### How
- Run Lean as a long-lived worker process per session
- Send incremental edits and parse diagnostics
- Expose `/session/start`, `/session/step`, `/session/clone`, `/session/close`

### Risks
- Most complex part of the stack
- requires robust session isolation and cleanup

---

## 2) Multi-tier model routing (cheap→expensive escalation) (HIGH impact, low-medium effort)
### What
Introduce a policy that escalates from cheap to expensive models based on failure type / progress.

### Why
- large cost reduction
- allows broad exploration cheaply and “heavy thinking” only when needed

### How
- Add multiple backends in Model Gateway
- Define escalation rules:
  - cheap model for initial attempts + simple repairs
  - expensive model only when stuck or for planning/refinement

### Risks
- more configuration + accounting
- inconsistent formats across providers

---

## 3) Proof search as “session beam” with cloning and rollback (HIGH impact)
### What
Formalize stepwise proving as a **beam search over proof states**, with reliable session cloning.

### Why
- branching at each step avoids dead-ends
- strong for problems where a few tactic choices matter

### How
- Maintain K sessions (proof states)
- Expand each with N candidate tactics
- Verify each, keep best K by progress heuristic
- Add loop detection via goal-state hashes

### Risks
- resource intensive; needs good budget manager
- requires robust goal-state parsing

---

## 4) Better progress heuristics: goal count + goal “difficulty” signals (HIGH impact)
### What
Improve branch scoring beyond “errors/no errors.”

### Why
- beam search lives or dies on scoring
- helps avoid wasting budget on unproductive branches

### How
Extract signals from Lean goal state:
- number of goals
- size of context (hypotheses count)
- presence of metavariables / typeclass goals
- whether goals are “simp closed” (e.g., `⊢ True`, `⊢ a = a`)
- detection of “stuck” patterns (same goal hash repeated)

### Risks
- parsing fragility
- avoid overfitting heuristics to one domain

---

## 5) Smarter retrieval: hybrid lexical + semantic + type-aware (HIGH impact, medium effort)
### What
Upgrade mathlib retrieval from regex+FTS to:
- richer signatures (elaborated types)
- semantic search embeddings
- typeclass-aware retrieval (e.g., `[Ring α]`, `[LinearOrderedRing α]`)

### Why
- missing lemma names/imports are a top failure mode
- type-aware retrieval reduces hallucinated lemmas

### How
- Build a second index by asking Lean to print declaration types (offline)
- Store:
  - pretty-printed type
  - normalized “type tokens”
- Add embeddings later (optional)
- Query expansion with:
  - type tokens from goal
  - error tokens from Lean

### Risks
- offline index build may be slow
- embedding costs unless you self-host

---

## 6) Import management / minimization (MED impact, low effort)
### What
Stop importing `Mathlib` globally; instead use:
- `import Mathlib` for initial V0
- later: generate minimal imports based on retrieval + failures

### Why
- speeds compilation checks
- reduces namespace noise and ambiguity

### How
- Start with `Mathlib` but allow additional imports
- In later attempts, use top retrieved file paths to add `import Mathlib.X.Y`
- Optionally run a minimizer pass after success

### Risks
- import errors create false negatives
- easiest path is to keep `import Mathlib` until interactive Lean mode is stable

---

## 7) Error-driven repair policies (MED impact, low effort)
### What
Use different repair prompts and strategies based on error_class.

### Why
A “one-size repair prompt” wastes tokens and causes thrashing.

### How
Create per-class repair handlers:
- `unknown_identifier`: call retriever, propose lemma names/imports
- `type_mismatch`: suggest rewriting statement, adjusting binders/types
- `tactic_failed`: suggest alternative tactic families (simp/cases/induction)
- `unsolved_goals`: ask for finishing tactics (`aesop`, `linarith`, `omega`, `ring`)
- `timeout`: add lemmas / split cases / reduce simp set

### Risks
- prompt sprawl; keep templates versioned and tested

---

## 8) Add a “toolbox” tactic set and auto-try pass (MED impact, very low effort)
### What
Before LLM calls, run a deterministic set of tactic attempts:
- `by simp`
- `by aesop`
- `by omega` (if available)
- `by nlinarith`
- `by ring`
- `by decide` (for decidable goals)

### Why
Catches a surprising fraction of goals cheaply; reduces model calls.

### How
- Add a small fixed candidate list at round 0
- Verify these first

### Risks
- might require extra imports / tactics availability
- can be slow if `aesop` blows up (cap timeouts)

---

## 9) Lemma blueprint “edit operators” (HIGH impact once planning is used)
### What
Make refinement systematic using explicit edit operations.

### Why
Planning improves when refinement isn’t random.

### How
Define edits:
- weaken lemma (remove condition / change equality to ≤, etc.)
- strengthen lemma (add missing constraints)
- split lemma into two
- add bridging lemma
- reorder lemma schedule
- change strategy tag (induction ↔ cases, etc.)
Then ask the planner to output an edit instead of a full new blueprint.

### Risks
- requires stable blueprint schema and versioning
- edit application needs validation

---

## 10) Blueprint beam search (MED-HIGH impact, medium effort)
### What
Keep multiple alternative blueprints, not just one.

### Why
Many failures are “wrong plan” rather than “bad proving.”

### How
- Maintain K blueprints (K=2–5)
- Allocate budget proportionally to their promise
- Score by:
  - lemmas proved
  - progress on main theorem
  - cost so far
  - failure severity

### Risks
- cost blowup if you don’t budget tightly
- requires good blueprint scoring

---

## 11) Library of proved lemmas across jobs (MED impact, medium effort)
### What
Persist proved lemmas (not just within one proof job) and reuse them.

### Why
Strong for benchmark suites and repeated domains (Putnam-like).

### How
- Store lemma statement hash → proof
- On new job, retrieve similar statements and try reuse
- Validate reused lemma compiles in current environment

### Risks
- compatibility issues if environment changes
- “polluting” the library with overly specialized lemmas

---

## 12) Detect and avoid “proof of wrong theorem” (relevant when you later do NL→formal)
### What
Add semantic checks. For formal-only ATP this is less relevant, but if you later add autoformalization it becomes crucial.

### Why
LLMs can accidentally solve a different statement if you allow them to edit it.

### How
- For formal-only ATP: disallow edits to theorem statement unless explicit
- For NL→formal later:
  - back-translate formal statement and compare
  - cross-check with counterexample search (Python)
  - multi-judge consistency checks

### Risks
- LLM judges can be noisy; need careful evaluation

---

## 13) “Portfolio prover” mode: multiple tactic policies + multiple models (MED-HIGH impact)
### What
Run different proof policies in parallel and take first success.

### Why
Different models/prompt styles excel at different domains.

### How
- Configure multiple “prover profiles”:
  - simp-heavy
  - induction-first
  - algebraic tactics
  - aesop-heavy
- Run them in parallel with separate budgets

### Risks
- cost management complexity
- results harder to reproduce unless you log everything

---

## 14) Speed optimization: precompiled environments and warm workers (HIGH impact in production)
### What
Keep LeanWorker instances warm with loaded environment and cached compilation artifacts.

### Why
Compilation time dominates costs in batch mode.

### How
- persistent workers
- load mathlib once
- avoid re-building per request
- use GKE if Cloud Run cold starts dominate

### Risks
- state leaks across sessions unless isolated
- scaling and cleanup

---

## 15) Add “simp set management” (MED impact)
### What
Teach the prover to manage simp lemmas intentionally:
- `simp [L1, L2]`
- avoid `simp` explosions
- maintain a curated simp set per domain

### Why
`simp` is powerful but can loop/explode; control matters.

### How
- Prompt pattern: always include explicit simp lemmas from retrieval
- Add a simplifier budget (max steps/time)
- If simp fails, try targeted rewriting instead

### Risks
- needs good retrieval and lemma naming accuracy

---

## 16) Add a “proof refactoring” pass (LOW-MED impact, nice-to-have)
### What
After finding a proof, optionally compress/refactor:
- replace long sequences with a lemma
- reorder arguments
- minimize imports

### Why
Makes artifacts reusable and speeds future proofs.

### How
- run a “refactor agent” on the final proof
- re-verify with LeanWorker

### Risks
- can introduce regressions; must verify

---

## Recommended next improvements (if you want a shortlist)
If you want the top “next 5” after Milestone 6:
1) True interactive Lean sessions (or at least faster stepwise state extraction)
2) Multi-tier model routing (cheap→expensive) + portfolio prover profiles
3) Better retrieval (type-aware + error-token expansion)
4) Beam over proof states with goal-based scoring + loop detection
5) Blueprint edits + blueprint beam search

---
```
