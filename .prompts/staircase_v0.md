# Staircase V0: Minimal Single-Shot Gemini Prover

## Overview

This is the **simplest possible iteration** of the ATP system. It establishes the baseline by:
1. Making a single Gemini Flash API call with a curated system prompt
2. Writing the returned Lean code to a file
3. Compiling with Lean 4 / mathlib
4. Scoring on 20 PutnamBench problems

**No orchestration, no repair loops, no retrieval, no caching.** Just: prompt → generate → verify → score.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PutnamBench    │────▶│  Gemini Flash   │────▶│  Lean Verifier  │
│  (20 problems)  │     │  (single shot)  │     │  (lake build)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Score: X/20    │
                                                └─────────────────┘
```

---

## Components to Build

### 1. Problem Loader (`problems/`)
- Store 20 curated PutnamBench problems as JSON files
- Each problem contains:
  - `id`: unique identifier (e.g., `putnam_1990_a1`)
  - `statement`: the formal Lean 4 theorem statement
  - `imports`: required imports (default: `import Mathlib`)

### 2. Gemini Client (`src/llm.py`)
- Simple wrapper around Google's Gemini API
- Uses `gemini-2.0-flash` model
- Single function: `generate_proof(statement: str) -> str`
- Curated system prompt for Lean 4 proof generation

### 3. Lean Verifier (`src/verifier.py`)
- Creates a temporary Lean project with pinned mathlib
- Writes generated proof to a `.lean` file
- Runs `lake build` with timeout
- Returns: `{success: bool, error: str | None, time_ms: int}`

### 4. Benchmark Runner (`src/benchmark.py`)
- Loads all 20 problems
- For each problem:
  - Call Gemini to generate proof
  - Verify with Lean
  - Record result
- Output final score and detailed results

### 5. Lean Project Template (`lean_project/`)
- `lakefile.lean` pinned to specific mathlib version
- `lean-toolchain` pinned to specific Lean version
- Used as template for verification

---

## System Prompt (Curated)

```
You are a Lean 4 theorem prover. Given a theorem statement, produce a complete proof.

RULES:
1. Output ONLY the proof term or tactic block (starting with `by`)
2. Do NOT include the theorem statement, imports, or any explanation
3. Use tactics from Mathlib: simp, ring, linarith, nlinarith, omega, norm_num, field_simp, polyrith, aesop
4. For induction, use `induction n with ...` syntax
5. Keep proofs concise - prefer powerful automation tactics
6. If multiple goals, use `·` bullet points or `<;>` combinator

EXAMPLES:
Statement: ∀ n : ℕ, n + 0 = n
Proof: by simp

Statement: ∀ a b : ℝ, (a + b)^2 = a^2 + 2*a*b + b^2
Proof: by ring
```

---

## File Structure

```
atp0/
├── .prompts/           # Existing prompt docs
├── lean_project/       # Template Lean project
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── lake-manifest.json
├── problems/           # 20 PutnamBench problems
│   └── problems.json
├── src/
│   ├── __init__.py
│   ├── llm.py          # Gemini client
│   ├── verifier.py     # Lean verification
│   └── benchmark.py    # Main runner
├── results/            # Output directory
├── requirements.txt
└── run_benchmark.py    # Entry point
```

---

## Success Criteria

1. **Runnable**: `python run_benchmark.py` executes end-to-end
2. **Reproducible**: Same problems, same model, deterministic verification
3. **Scored**: Outputs `X/20` with detailed per-problem results
4. **Logged**: Saves all generated proofs and verification output

---

## Expected Baseline

For a single-shot approach with no repair:
- **Optimistic**: 3-5/20 (trivial problems solved by automation tactics)
- **Realistic**: 1-3/20 (Putnam problems are hard)
- **Pessimistic**: 0/20 (if problems require deep reasoning)

This establishes the baseline for measuring improvement in later iterations.

---

## Next Iteration (Staircase V1)

After V0, the next iteration will add:
- **N-shot sampling**: Generate 5 candidates, take first success
- **Basic error feedback**: If verification fails, show error to model for one retry

This document is complete. Proceed with implementation.
