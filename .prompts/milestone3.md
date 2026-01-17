# Milestone 3 — Mathlib Retriever + Retrieval-Augmented Proving (RAG)

## Goal
Add a **mathlib retrieval service** and integrate it into the Orchestrator so that proof generation and repair are **retrieval-augmented**. This improves:
- library alignment (correct lemma/definition names)
- import guidance
- repair quality when Lean reports unknown identifiers or typeclass synthesis failures

Milestone 3 keeps the proving loop from Milestone 2, but upgrades inputs to the Model Gateway with **retrieved context**.

**No training.** Retrieval is purely indexing + search + prompt augmentation.

---

## Dependencies
- Milestone 1: LeanWorker `/check`
- Milestone 2: Orchestrator `/prove` with Model Gateway and repair loop

---

## Scope (what to build)

### Deliverables
1. **Retriever service** with:
   - an offline **index builder**
   - an online query API `POST /search`
2. Index built from **mathlib4 source files** (fast to implement)
3. Orchestrator integration:
   - query retriever at the start of a proof job and on key failures
   - include retrieved declarations in proposer/repair prompts
4. Optional but recommended:
   - **import suggestion heuristic** (coarse)
   - query expansion using tokens from Lean errors
5. Minimal benchmarking demonstrating improvement on cases where the dummy / baseline fails due to missing lemma names

---

## Non-goals (Milestone 3)
- No embeddings required in V0 (FTS keyword search is enough)
- No interactive Lean REPL
- No lemma decomposition (Milestone 4)
- No sophisticated premise selection model

---

## Retriever: Data and Indexing

### Data source
Use **mathlib4 source** (the `Mathlib/` directory within the pinned mathlib commit) from the LeanWorker image or a local clone pinned to the same commit.

### What to index (V0)
Index declarations:
- `theorem`, `lemma`, `def`, `class`, `structure`, `instance`, `abbrev`

For each declaration, store:
- `name` (fully qualified if available)
- `kind` (theorem/lemma/def/...)
- `signature` (best-effort single-line type statement)
- `doc` (optional: preceding docstring/comment block if parseable)
- `file_path`
- `line_start` (optional)
- `namespace` (optional)

> V0 parse can be heuristic; you are not building a Lean parser.

---

## Index builder (offline)

### Output format
Use SQLite with FTS5 for fast text search.

Tables:
- `decls(id INTEGER PRIMARY KEY, name TEXT, kind TEXT, signature TEXT, doc TEXT, file_path TEXT, line_start INT)`
- `decls_fts` (FTS5 virtual table) over `name`, `signature`, `doc`

Recommended: keep `name` normalized and also store a lowercased version for search.

### Build procedure (must be deterministic)
1. Walk `Mathlib/**/*.lean`
2. Extract declaration blocks using regex heuristics:
   - detect lines starting with `theorem|lemma|def|class|structure|instance|abbrev`
   - capture name token immediately after keyword
   - capture until `:=` or `:` line end (signature line)
3. Store into SQLite
4. Build FTS index

Deliverable:
- `retriever/build_index.py`
- produces `retriever/index/mathlib.sqlite`

### Heuristic parsing notes
- Many decl signatures span multiple lines. In V0:
  - capture first line after keyword
  - optionally append subsequent lines until you hit `:=` or blank line (limit to ~10 lines)
- Strip tactics/proofs; we only need statement/type signature.

---

## Retriever service (online)

### Endpoint: `POST /search`
#### Request
```json
{
  "query": "Nat.add_comm",
  "k": 10,
  "filters": {
    "kind": ["theorem", "lemma"]
  }
}
Response
json
Copy code
{
  "query": "Nat.add_comm",
  "results": [
    {
      "name": "Nat.add_comm",
      "kind": "theorem",
      "signature": "theorem Nat.add_comm (a b : Nat) : a + b = b + a := ...",
      "doc": "optional",
      "file_path": "Mathlib/Data/Nat/Basic.lean",
      "line_start": 123
    }
  ]
}
Search implementation
Run FTS match on:

name boosted highest

signature medium

doc low

Provide a simple rank score from SQLite FTS.

Additional endpoint (optional)
GET /stats returns indexed decl count + commit hash.

Orchestrator integration (RAG)
When to query the retriever
At job start:

build query from theorem statement tokens + types + operators

On failure (repair loop):

if error_class in {unknown_identifier, type_mismatch}:

extract key tokens from Lean error message (e.g., missing constant name, typeclass name)

query retriever with those tokens

if tactic fails:

query for relevant lemmas using goal keywords (e.g., “dvd”, “coprime”, “Finset”)

Query construction (V0 rules)
Extract tokens from theorem statement:

identifiers: Nat, Int, Finset, Group, Set, etc.

operators: +, *, ≤, ∣, ∈ mapped to keywords (add, mul, le, dvd, mem)

Add any unknown constant identifiers from Lean errors directly.

Retrieved context formatting (important)
Insert into prompts as a stable block:

yaml
Copy code
# Retrieved Mathlib Declarations (top 10)
- Nat.add_comm : ∀ a b : Nat, a + b = b + a
  file: Mathlib/Data/Nat/Basic.lean
- ...
Constraints:

Limit total retrieved context tokens (e.g., max 1500 chars)

Prefer signatures over docs

Include exact lemma names

Prompt updates (Model Gateway)
Update both propose + repair prompts to include a retrieved section.

Proposer prompt must instruct:
Use the retrieved lemma names exactly

Don’t invent lemma names; if needed, call simp with bracketed lemmas:

simp [Nat.add_comm]

Output only Lean proof block starting with by

Repair prompt must:
include Lean error excerpt

include retrieved results (especially when unknown identifiers occur)

instruct: “Fix missing lemma names/imports by using retrieved declarations; avoid invented names.”

Import suggestion (optional but recommended)
V0 heuristic
If retrieved results include file paths, you can propose additional imports:

Take top 1–3 file paths from retrieved results

Convert path to import string (roughly):

Mathlib/Data/Nat/Basic.lean → Mathlib.Data.Nat.Basic

Add to imports list for the next attempts, but keep import Mathlib always.

This is optional; many lemmas are available from import Mathlib anyway. But on some setups, it helps.

Caching (upgrade)
Extend Milestone 2 cache to include:

retrieval results per (stmt_hash, query_hash)

store extracted tokens from errors to avoid repeated parsing

Acceptance tests (Milestone 3)
Test structure
Create at least 3 benchmark cases that require correct lemma naming:

A theorem where by simp fails but by simpa [Nat.add_comm] works

A theorem involving sets/finsets where retrieval provides the right lemma name

A failure case where unknown identifier occurs; verify retriever is queried and results appear in the prompt logs

Required assertions
Orchestrator logs must show:

“retrieval query” at job start

additional retrieval queries on unknown_identifier failures

Prover prompt (for dummy backend) can be a no-op, but in real backends this must be wired.

For at least one case, show improved success with retrieval-enabled candidate set (e.g., model proposes simp [<retrieved lemma>]).

Definition of Done
build_index.py produces a deterministic SQLite index for the pinned mathlib commit

Retriever service can search and return top-k lemma signatures

Orchestrator:

calls retriever at job start

calls retriever on at least unknown_identifier failures

includes retrieved context in propose/repair prompts

At least one benchmark shows improved solve rate with retrieval enabled

All services run locally together (docker-compose optional)

Notes for Milestone 4+
Retriever becomes a key part of planning (lemma decomposition will query it heavily)

Future upgrades:

add embeddings for semantic search

index elaborated types by calling Lean to print declaration types

premise selection heuristics based on goal type unification

makefile
Copy code
::contentReference[oaicite:0]{index=0}