#!/usr/bin/env python3
"""V2 benchmark: Multi-round verification-guided repair + caching."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file from script directory
load_dotenv(Path(__file__).parent / ".env")

LEAN_PROJECT = Path(__file__).parent / "lean_project"
KIMINA_URL = "https://lean.cajal.org/api/check"
TIMEOUT = 120
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

PROPOSE_PROMPT = """You are a Lean 4 theorem prover.

TASK: Produce exactly {k} DIFFERENT Lean proof candidates for the theorem below.

RULES:
- Output ONLY valid JSON: {{"candidates":[{{"strategy":"<tag>","proof":"<proof starting with by>"}}, ...]}}
- Each proof MUST start with `by`
- Do NOT include imports, theorem headers, or explanations
- Do NOT use `sorry`
- Candidates must use DIFFERENT strategies

STRATEGY TAGS: simp, aesop, induction, cases, algebra, have_chain, calc, inequalities

THEOREM:
{theorem}

K = {k}"""

REPAIR_PROMPT_V2 = """You are repairing a Lean 4 proof.

THEOREM:
{theorem}

FAILED PROOF:
{failed_proof}

LEAN ERROR:
{error}

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

R = {r}"""

client = None
print_lock = Lock()

def log(msg: str, end="\n", flush=True):
    """Thread-safe print."""
    with print_lock:
        print(msg, end=end, flush=flush)

# ============== Error Classification ==============

def classify_error(error_excerpt: str) -> str:
    """Classify error type from excerpt."""
    if not error_excerpt:
        return "other"
    error_lower = error_excerpt.lower()

    if "unknown identifier" in error_lower or "unknown constant" in error_lower:
        return "unknown_identifier"
    if "type mismatch" in error_lower or "has type" in error_lower:
        return "type_mismatch"
    if "unsolved goals" in error_lower:
        return "unsolved_goals"
    if "tactic" in error_lower and "failed" in error_lower:
        return "tactic_failed"
    if "expected" in error_lower and ("token" in error_lower or "command" in error_lower):
        return "parse_error"
    if "sorry" in error_lower:
        return "sorry"
    return "other"

# ============== Proof Normalization & Caching ==============

def normalize_proof(proof: str) -> str:
    """Normalize proof for deduplication."""
    # Remove extra whitespace, normalize indentation
    lines = [line.strip() for line in proof.strip().split("\n") if line.strip()]
    return "\n".join(lines)

def proof_hash(proof: str) -> str:
    """Hash normalized proof for caching."""
    return hashlib.md5(normalize_proof(proof).encode()).hexdigest()[:12]

def clean_proof(proof: str) -> str:
    """Clean proof text: remove markdown, ensure starts with 'by'."""
    if "```" in proof:
        proof = "\n".join(l for l in proof.split("\n") if "```" not in l)
    proof = proof.strip()
    if not proof.startswith("by"):
        proof = f"by\n  {proof}"
    return proof

def dedupe_proofs(proofs: list[str], seen: set[str]) -> list[str]:
    """Remove duplicate proofs, update seen set."""
    unique = []
    for p in proofs:
        cleaned = clean_proof(p)
        h = proof_hash(cleaned)
        if h not in seen:
            seen.add(h)
            unique.append(cleaned)
    return unique

# ============== LLM Generation ==============

def generate_proofs(theorem: str, k: int) -> list[str]:
    """Generate K diverse proof candidates."""
    prompt = PROPOSE_PROMPT.format(theorem=theorem, k=k)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            data = json.loads(text)
            proofs = [c["proof"].strip() for c in data.get("candidates", [])]
            return [p if p.startswith("by") else f"by\n  {p}" for p in proofs if p]
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait_time = 20 * (attempt + 1)
                log(f"(rate limited, waiting {wait_time}s)", end=" ")
                time.sleep(wait_time)
                continue
            log(f"(generation error: {str(e)[:50]})", end=" ")
            if attempt == 2:
                return []
            continue
    return []

def generate_repairs_v2(theorem: str, failed_proof: str, error: str, error_class: str, r: int) -> list[str]:
    """Generate R repair candidates with error-class-aware prompt."""
    prompt = REPAIR_PROMPT_V2.format(
        theorem=theorem,
        failed_proof=failed_proof,
        error=error[:500],
        error_class=error_class,
        r=r
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            data = json.loads(text)
            proofs = [c["proof"].strip() for c in data.get("repairs", [])]
            return [p if p.startswith("by") else f"by\n  {p}" for p in proofs if p]
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait_time = 20 * (attempt + 1)
                log(f"(rate limited, waiting {wait_time}s)", end=" ")
                time.sleep(wait_time)
                continue
            return []
    return []

# ============== Verification ==============

def verify_batch_server(header: str, theorem: str, proofs: list[str], timeout: int = 60) -> list[dict]:
    """Verify multiple proofs in a single batch request to Kimina server."""
    if not proofs:
        return []

    snippets = []
    cleaned_proofs = []
    for i, proof in enumerate(proofs):
        cleaned = clean_proof(proof)
        cleaned_proofs.append(cleaned)
        lean_code = f"{header}\n\n{theorem} :=\n{cleaned}\n"
        snippets.append({"id": f"proof_{i}", "code": lean_code})

    start = time.time()
    try:
        response = requests.post(
            KIMINA_URL,
            json={"snippets": snippets, "timeout": timeout, "reuse": True},
            timeout=timeout + 10
        )
        response.raise_for_status()
        data = response.json()
        elapsed = int((time.time() - start) * 1000)

        results = []
        for i, res in enumerate(data.get("results", [])):
            error = res.get("error")
            resp = res.get("response") or {}
            messages = resp.get("messages", [])
            error_msgs = [m for m in messages if m.get("severity") == "error"]
            has_sorry = "sorry" in str(resp).lower()

            success = error is None and not error_msgs and not has_sorry

            error_excerpt = None
            if not success:
                if error:
                    error_excerpt = error[:300]
                elif error_msgs:
                    error_excerpt = "\n".join(m.get("data", "")[:100] for m in error_msgs[:3])
                elif has_sorry:
                    error_excerpt = "uses sorry"
                else:
                    error_excerpt = "unknown error"

            results.append({
                "idx": i,
                "success": success,
                "time_ms": int(res.get("time", 0) * 1000) or elapsed // len(proofs),
                "proof": cleaned_proofs[i],
                "proof_hash": proof_hash(cleaned_proofs[i]),
                "error_excerpt": error_excerpt,
                "error_class": classify_error(error_excerpt) if not success else None,
                "output": str(res)[:500]
            })
        return results
    except Exception as e:
        return [
            {"idx": i, "success": False, "time_ms": 0, "proof": clean_proof(p),
             "proof_hash": proof_hash(clean_proof(p)),
             "error_excerpt": f"Server error: {e}", "error_class": "other", "output": str(e)}
            for i, p in enumerate(proofs)
        ]

def verify_proof_local(header: str, theorem: str, proof: str, idx: int, prob_idx: int = 0) -> dict:
    """Verify a single proof locally."""
    proof = clean_proof(proof)
    lean_code = f"{header}\n\n{theorem} :=\n{proof}\n"
    verify_file = LEAN_PROJECT / f"Verify_p{prob_idx}_c{idx}.lean"
    verify_file.write_text(lean_code)

    start = time.time()
    try:
        result = subprocess.run(
            ["lake", "env", "lean", str(verify_file)],
            cwd=LEAN_PROJECT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        elapsed = int((time.time() - start) * 1000)
        output = result.stdout + result.stderr
        success = result.returncode == 0 and "uses 'sorry'" not in output

        error_excerpt = None
        if not success:
            lines = output.split("\n")
            error_lines = [l for l in lines if "error" in l.lower()][:5]
            error_excerpt = "\n".join(error_lines) if error_lines else output[:300]

        return {
            "idx": idx,
            "success": success,
            "time_ms": elapsed,
            "proof": proof,
            "proof_hash": proof_hash(proof),
            "error_excerpt": error_excerpt,
            "error_class": classify_error(error_excerpt) if not success else None,
            "output": output[:500]
        }
    except subprocess.TimeoutExpired:
        return {"idx": idx, "success": False, "time_ms": TIMEOUT*1000, "proof": proof,
                "proof_hash": proof_hash(proof), "error_excerpt": "TIMEOUT", "error_class": "other", "output": "TIMEOUT"}
    except Exception as e:
        return {"idx": idx, "success": False, "time_ms": 0, "proof": proof,
                "proof_hash": proof_hash(proof), "error_excerpt": str(e), "error_class": "other", "output": str(e)}
    finally:
        verify_file.unlink(missing_ok=True)

def verify_many_local(header: str, theorem: str, proofs: list[str], max_parallel: int, executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
    """Verify multiple proofs in parallel locally."""
    if not proofs:
        return []
    results = []
    futures = {executor.submit(verify_proof_local, header, theorem, p, i, prob_idx): i for i, p in enumerate(proofs)}
    for future in as_completed(futures):
        results.append(future.result())
    return results

def verify_proofs(header: str, theorem: str, proofs: list[str], args, executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
    """Verify proofs using server (default) or local."""
    if args.local:
        return verify_many_local(header, theorem, proofs, args.max_parallel, executor, prob_idx)
    else:
        return verify_batch_server(header, theorem, proofs, timeout=args.timeout)

# ============== Failure Selection ==============

ERROR_PRIORITY = {
    "unsolved_goals": 0,
    "tactic_failed": 1,
    "type_mismatch": 2,
    "unknown_identifier": 3,
    "parse_error": 4,
    "sorry": 5,
    "other": 6
}

def select_best_failures(attempts: list[dict], top_n: int) -> list[dict]:
    """Select top failures, prioritizing by error class and diversity."""
    failures = [a for a in attempts if not a["success"]]
    if not failures:
        return []

    # Sort by error priority, then by shortest error
    failures.sort(key=lambda x: (
        ERROR_PRIORITY.get(x.get("error_class", "other"), 6),
        len(x.get("error_excerpt", "") or "")
    ))

    # Select diverse failures (different error classes if possible)
    selected = []
    seen_classes = set()
    for f in failures:
        ec = f.get("error_class", "other")
        if ec not in seen_classes:
            selected.append(f)
            seen_classes.add(ec)
            if len(selected) >= top_n:
                break

    # Fill remaining slots if needed
    for f in failures:
        if f not in selected:
            selected.append(f)
            if len(selected) >= top_n:
                break

    return selected[:top_n]

# ============== Main Problem Processing ==============

def process_problem_v2(prob: dict, prob_idx: int, args, executor: ThreadPoolExecutor) -> dict:
    """Process a single problem with multi-round repair loop."""
    prob_id = prob["id"]
    header = prob["header"]
    theorem = prob["theorem"]

    seen_proofs: set[str] = set()
    all_attempts: list[dict] = []
    total_verifications = 0
    error_class_counts: dict[str, int] = {}

    # Round 0: Propose
    log(f"  [R0] Generating {args.k} candidates...", end=" ")
    proofs = generate_proofs(theorem, args.k)
    proofs = dedupe_proofs(proofs, seen_proofs)
    log(f"got {len(proofs)} unique")

    if proofs:
        log(f"  [R0] Verifying...", end=" ")
        results = verify_proofs(header, theorem, proofs, args, executor, prob_idx)
        for r in results:
            r["round"] = 0
        all_attempts.extend(results)
        total_verifications += len(results)

        # Count error classes
        for r in results:
            if not r["success"] and r.get("error_class"):
                error_class_counts[r["error_class"]] = error_class_counts.get(r["error_class"], 0) + 1

        successes = [r for r in results if r["success"]]
        if successes:
            winner = successes[0]
            log(f"✓ SOLVED R0 (candidate {winner['idx']}, {winner['time_ms']}ms)")
            return {
                "id": prob_id, "success": True, "winning_round": 0,
                "rounds_used": 1, "verifications_used": total_verifications,
                "proof": winner["proof"], "winning_idx": winner["idx"],
                "pass_at_1": winner["idx"] == 0,
                "all_attempts": all_attempts,
                "error_class_counts": error_class_counts
            }
        log(f"all {len(results)} failed")

    # Repair rounds
    for round_num in range(1, args.max_rounds + 1):
        if total_verifications >= args.max_verifications:
            log(f"  Budget exhausted ({total_verifications}/{args.max_verifications})")
            break

        failures = select_best_failures(all_attempts, args.top_failures)
        if not failures:
            break

        round_repairs = []
        for fi, failure in enumerate(failures):
            error_class = failure.get("error_class", "other")
            log(f"  [R{round_num}] Repairing failure (class={error_class})...", end=" ")

            repairs = generate_repairs_v2(
                theorem, failure["proof"],
                failure.get("error_excerpt", ""), error_class,
                args.repairs_per_failure
            )
            repairs = dedupe_proofs(repairs, seen_proofs)
            log(f"got {len(repairs)} unique")

            if repairs:
                round_repairs.extend(repairs)

        if round_repairs:
            remaining_budget = args.max_verifications - total_verifications
            round_repairs = round_repairs[:remaining_budget]

            log(f"  [R{round_num}] Verifying {len(round_repairs)} repairs...", end=" ")
            results = verify_proofs(header, theorem, round_repairs, args, executor, prob_idx + round_num * 100)
            for r in results:
                r["round"] = round_num
            all_attempts.extend(results)
            total_verifications += len(results)

            # Count error classes
            for r in results:
                if not r["success"] and r.get("error_class"):
                    error_class_counts[r["error_class"]] = error_class_counts.get(r["error_class"], 0) + 1

            successes = [r for r in results if r["success"]]
            if successes:
                winner = successes[0]
                log(f"✓ SOLVED R{round_num} ({winner['time_ms']}ms)")
                return {
                    "id": prob_id, "success": True, "winning_round": round_num,
                    "rounds_used": round_num + 1, "verifications_used": total_verifications,
                    "proof": winner["proof"],
                    "pass_at_1": False,
                    "all_attempts": all_attempts,
                    "error_class_counts": error_class_counts
                }
            log(f"all failed")

    log(f"  ✗ FAILED after {total_verifications} verifications")
    return {
        "id": prob_id, "success": False,
        "rounds_used": min(args.max_rounds + 1, round_num + 1) if 'round_num' in dir() else 1,
        "verifications_used": total_verifications,
        "pass_at_1": False,
        "all_attempts": all_attempts,
        "error_class_counts": error_class_counts
    }

# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="V2 benchmark: Multi-round repair with caching")
    parser.add_argument("--k", type=int, default=3, help="Initial candidates to generate")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max repair rounds")
    parser.add_argument("--top-failures", type=int, default=2, help="Failures to repair per round")
    parser.add_argument("--repairs-per-failure", type=int, default=3, help="Repairs per failure")
    parser.add_argument("--max-verifications", type=int, default=50, help="Total verification budget per problem")
    parser.add_argument("--timeout", type=int, default=60, help="Server timeout per batch")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel local verifications")
    parser.add_argument("--problems", type=str, default="problems.json", help="Problem file")
    parser.add_argument("--local", action="store_true", help="Use local Lean instead of server")
    args = parser.parse_args()

    global client
    # Prefer GEMINI_API_KEY, fall back to GOOGLE_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY or GOOGLE_API_KEY")
        return
    # Set both to avoid SDK warning and ensure correct key is used
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    # Configure client with longer timeout and automatic retries
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120000,  # 120s timeout
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=30.0,
                exp_base=2.0,
                http_status_codes=[429, 500, 502, 503, 504]
            )
        )
    )

    problems_path = Path(__file__).parent / "problems" / args.problems
    with open(problems_path) as f:
        problems = json.load(f)

    results = []
    solved = 0
    pass_at_1 = 0
    total_verifications = 0
    total_rounds = 0
    solved_rounds = []
    global_error_counts: dict[str, int] = {}

    mode = "local" if args.local else "server"
    print(f"V2 Benchmark: K={args.k}, max_rounds={args.max_rounds}, mode={mode}, problems={args.problems}")
    print(f"Budget: {args.max_verifications} verifications/problem, {args.top_failures} failures × {args.repairs_per_failure} repairs/round")
    print(f"Running on {len(problems)} problems...\n")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        for i, prob in enumerate(problems):
            log(f"[{i+1}/{len(problems)}] {prob['id']}")
            result = process_problem_v2(prob, i, args, executor)
            results.append(result)

            if result["success"]:
                solved += 1
                solved_rounds.append(result["winning_round"])
            if result.get("pass_at_1"):
                pass_at_1 += 1
            total_verifications += result["verifications_used"]
            total_rounds += result["rounds_used"]

            # Aggregate error counts
            for ec, count in result.get("error_class_counts", {}).items():
                global_error_counts[ec] = global_error_counts.get(ec, 0) + count

    # Summary
    print(f"\n{'='*50}")
    print(f"SCORE: {solved}/{len(problems)}")
    print(f"pass@1: {pass_at_1}/{len(problems)}")
    print(f"pass@{args.k} (R0): {len([r for r in results if r.get('winning_round') == 0])}/{len(problems)}")
    print(f"repair wins: {len([r for r in results if r.get('winning_round', -1) > 0])}")
    print(f"mean verifications/problem: {total_verifications / len(problems):.1f}")
    if solved_rounds:
        print(f"mean rounds (solved): {sum(solved_rounds) / len(solved_rounds):.1f}")
    print(f"error classes: {global_error_counts}")
    print(f"{'='*50}")

    # Save
    prob_tag = args.problems.replace(".json", "").replace("problems_", "").replace("problems", "default")
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    timestamp_readable = now.strftime("%y%m%d-%H%M")
    results_path = Path(__file__).parent / "results" / f"v2_{prob_tag}_k{args.k}_r{args.max_rounds}_{timestamp_readable}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S PST"),
            "config": {
                "version": "v2", "model": MODEL,
                "k": args.k, "max_rounds": args.max_rounds,
                "top_failures": args.top_failures, "repairs_per_failure": args.repairs_per_failure,
                "max_verifications": args.max_verifications, "timeout": args.timeout,
                "mode": mode, "problems": args.problems
            },
            "score": f"{solved}/{len(problems)}",
            "pass_at_1": pass_at_1,
            "pass_at_k_r0": len([r for r in results if r.get("winning_round") == 0]),
            "repair_wins": len([r for r in results if r.get("winning_round", -1) > 0]),
            "mean_verifications": total_verifications / len(problems),
            "error_class_histogram": global_error_counts,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
