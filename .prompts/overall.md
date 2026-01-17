# Agent Context: Lean 4 ATP System (Aleph-like, LLM-agnostic) — V0 Build

This repository is building a **Lean 4 Automated Theorem Proving (ATP)** system inspired by modern “verification-guided” agents (e.g., Plan → Prove → Refine loops) and publicly described systems like “Aleph”. The core idea is:

> **LLMs propose proofs; Lean verifies them.**  
> **Verification is the filter; orchestration/search is the engine.**

The system is **LLM-agnostic** (works with any model backend via a gateway) and **does no training in V0**. Intelligence comes from: (1) candidate generation, (2) Lean verification feedback, (3) retrieval over mathlib, (4) search/orchestration, (5) caching and budgets.

This file exists to give any coding agent a complete overview so it can implement milestones without re-litigating design decisions.

---

## 1) Project Goal

### Primary objective
Build a service that accepts a **formal Lean 4 theorem statement** and attempts to produce a **machine-verified Lean 4 proof**.

### V0 philosophy
- **No training** (no finetuning / RL / DPO).
- Lean 4 is the only proof oracle.
- Use an **agentic orchestration loop**:
  - **Plan:** propose lemma decomposition / proof blueprint
  - **Prove:** generate candidate proofs
  - **Refine:** use Lean feedback to repair / backtrack / restructure
- Use **retrieval** over mathlib to align to library names/idioms.
- Use **parallel sampling + beam search** and **aggressive caching** for cost efficiency.

---

## 2) System Architecture Overview (Target)

### Services (target, may start locally then deploy to GCP)
1. **LeanWorker** (Verifier service)  
   - Runs Lean 4 with pinned toolchain + pinned mathlib
   - Accepts theorem statement + proof candidate
   - Returns ok/fail + structured diagnostics
   - This is Milestone 1 (must exist before anything else).

2. **Orchestrator** (Controller / API)  
   - Implements Plan → Prove → Refine loop
   - Schedules many proof attempts, collects Lean feedback, revises plans

3. **Model Gateway** (LLM-agnostic)  
   - Unified interface for any provider (OpenAI/Gemini/DeepSeek/local)
   - Provides planner/prover/repair calls with JSON output constraints

4. **Retriever** (mathlib search)  
   - Indexes mathlib declarations (names, signatures, docstrings, file paths)
   - Provides relevant lemma suggestions for planner/prover prompts

5. **Cache / Artifact store**  
   - Memoizes proof attempts and results, keyed by normalized goal hash
   - Stores winning proofs and logs

---

## 3) Non-Goals (V0)

- No model training, finetuning, RL, or learned value function
- No interactive IDE integration (Lean editor protocol) in V0
- No formalization-from-natural-language in V0 (input is already formal)
- No attempt to guarantee semantic alignment of an informal spec (that comes later)
- No heavy distributed scheduling required initially (can be single machine + parallel processes)

---

## 4) Key Design Decisions (Fixed for V0)

These decisions are locked to avoid analysis paralysis. If later milestones revise them, they will do so explicitly.

### A) Lean verification mode
- **V0 uses batch checking**:
  - Generate a temp Lean file `Main.lean`
  - Run `lake env lean Main.lean` under a pinned Lake project (mathlib pinned)
  - Parse stdout/stderr into structured diagnostics
- **Timeouts are mandatory** (process killed on timeout)

> Later: upgrade to true interactive goal-state REPL (Lean server protocol) for higher performance.

### B) Proof generation style (later milestones)
- Prefer **short tactic blocks** with frequent verification (not monolithic long proofs)
- “Stepwise tactic REPL mode” is planned but not required in Milestone 1

### C) Search strategy (later milestones)
- Use **verification-guided beam search** and local repair loops
- Use lemma decomposition (“blueprints”) but keep the first version simple

### D) Retrieval (later milestones)
- Use a simple, reliable index (SQLite FTS over mathlib sources) first
- Insert retrieved lemma signatures into prompts

### E) Deployment
- Target: **GCP**
  - Cloud Run for stateless services (Orchestrator / Retriever / Gateway)
  - LeanWorker pool on Cloud Run or GKE depending on throughput
  - GCS for artifacts, Firestore/Redis for caching later

---

## 5) LeanWorker (Milestone 1) — Core Contract

LeanWorker is the first required component. Everything else depends on it.

### Purpose
Given:
- imports
- optional prelude/decls
- theorem name + theorem statement
- proof candidate (`by ...`)

Return:
- ok/fail (Lean elaboration success)
- structured diagnostics (errors/warnings with line/col)
- stdout/stderr
- time_ms, exit_code
- echo of the generated Lean file (for debugging/artifacts)

### Template constraints
Lean file is deterministic:

1) Header comment with job_id
2) Imports
3) Options
4) extra_prelude
5) decls
6) theorem stub:
   - If `proof` starts with `by`, use directly
   - Else wrap as:
     ```
     theorem T : <stmt> := by
       <proof>
     ```

### Minimal endpoints
- `POST /check`
- `GET /healthz`
- `GET /version`

### Required behavior
- Must run under strict timeout
- Must not allow arbitrary filesystem access beyond temp dir
- Must be pinned to fixed Lean toolchain + mathlib commit

---

## 6) Repository Conventions

### Suggested layout (may evolve)

aleph_v0/
lean_worker/ # Milestone 1 focus
app/ # FastAPI server + Lean runner + template
LeanProject/ # lakefile.lean + lean-toolchain pinned to mathlib
Dockerfile
requirements.txt
orchestrator/ # later
retriever/ # later
shared/ # later
benchmarks/ # later
infra/ # later (terraform/k8s)

### Logging and artifacts
- Every check returns the exact generated Lean file for reproducibility
- Later orchestration will store artifacts in GCS (V0 can store locally)

---

## 7) Quality Bar and “Definition of Done”

For each milestone, the agent must:
1) Implement exactly what the milestone doc requests (no scope creep).
2) Provide a runnable setup (Docker build/run commands).
3) Provide at least a minimal smoke test.
4) Document any assumptions made.

For Milestone 1 specifically:
- `docker build` succeeds
- `docker run` starts service on port 8080
- `/healthz` returns ok
- `/check` passes at least:
  - a trivial proof (e.g., `∀ n : Nat, n = n`)
  - a failure case (unknown identifier) with correct error classification

---

## 8) How to Work With This Repo (Agent Instructions)

### General rules
- Don’t change earlier milestone specs unless a later milestone doc explicitly instructs it.
- If something is unclear, make the **minimal reasonable choice** and document it in-code.
- Keep interfaces stable and simple.
- Prefer deterministic behavior and reproducibility over micro-optimizations.

### Workflow
The repository will contain:
- this `AGENT_CONTEXT.md`
- separate milestone docs added over time (e.g., `MILESTONE_01_LEANWORKER.md`, `MILESTONE_02_ORCHESTRATOR_V0.md`, ...)

The agent should implement milestones sequentially.

---

## 9) Notes on Aleph-Like Behavior (Context Only)

The target architecture follows a common high-performing recipe:
- **verification-guided search**
- **planner + prover + repair** roles (all via LLM gateway)
- **parallel candidate generation**
- **persistent verifier workers**
- **heavy caching and budget control**

V0 begins by building the verifier substrate (LeanWorker). Everything else comes after.