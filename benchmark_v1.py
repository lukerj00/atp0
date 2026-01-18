#!/usr/bin/env python3
"""V1 benchmark: pass@K with strategy diversity + 1-step repair."""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from google import genai

LEAN_PROJECT = Path(__file__).parent / "lean_project"
TIMEOUT = 120

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

REPAIR_PROMPT = """You are repairing Lean 4 proofs.

THEOREM:
{theorem}

FAILED PROOF:
{failed_proof}

LEAN ERROR:
{error}

TASK: Produce exactly {r} DIFFERENT repaired proof candidates.

RULES:
- Output ONLY valid JSON: {{"repairs":[{{"strategy":"<tag>","proof":"<proof starting with by>"}}, ...]}}
- Each proof MUST start with `by`
- Do NOT use `sorry`
- Try different approaches

R = {r}"""

client = None
print_lock = Lock()

def log(msg: str, end="\n", flush=True):
    """Thread-safe print."""
    with print_lock:
        print(msg, end=end, flush=flush)

def generate_proofs(theorem: str, k: int) -> list[str]:
    """Generate K diverse proof candidates."""
    prompt = PROPOSE_PROMPT.format(theorem=theorem, k=k)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"temperature": 0.7}
            )
            text = response.text.strip()
            # Extract JSON from markdown code blocks
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            data = json.loads(text)
            proofs = [c["proof"].strip() for c in data.get("candidates", [])]
            return [p if p.startswith("by") else f"by\n  {p}" for p in proofs if p]
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            if attempt == 2:
                return []
            continue
    return []

def generate_repairs(theorem: str, failed_proof: str, error: str, r: int) -> list[str]:
    """Generate R repair candidates."""
    prompt = REPAIR_PROMPT.format(theorem=theorem, failed_proof=failed_proof, error=error[:500], r=r)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"temperature": 0.8}
            )
            text = response.text.strip()
            # Extract JSON from markdown code blocks
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            data = json.loads(text)
            proofs = [c["proof"].strip() for c in data.get("repairs", [])]
            return [p if p.startswith("by") else f"by\n  {p}" for p in proofs if p]
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            return []
    return []

def verify_proof(header: str, theorem: str, proof: str, idx: int, prob_idx: int = 0) -> dict:
    """Verify a single proof, return result dict."""
    if "```" in proof:
        proof = "\n".join(l for l in proof.split("\n") if "```" not in l)
    proof = proof.strip()
    if not proof.startswith("by"):
        proof = f"by\n  {proof}"

    lean_code = f"{header}\n\n{theorem} :=\n{proof}\n"
    # Use unique file per problem+candidate to avoid conflicts in parallel
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
        # Reject proofs that use sorry (Lean only warns, doesn't error)
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
            "error_excerpt": error_excerpt,
            "output": output[:500]
        }
    except subprocess.TimeoutExpired:
        return {"idx": idx, "success": False, "time_ms": TIMEOUT*1000, "proof": proof, "error_excerpt": "TIMEOUT", "output": "TIMEOUT"}
    except Exception as e:
        return {"idx": idx, "success": False, "time_ms": 0, "proof": proof, "error_excerpt": str(e), "output": str(e)}
    finally:
        verify_file.unlink(missing_ok=True)

def verify_many(header: str, theorem: str, proofs: list[str], max_parallel: int, executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
    """Verify multiple proofs in parallel using shared executor."""
    results = []
    futures = {executor.submit(verify_proof, header, theorem, p, i, prob_idx): i for i, p in enumerate(proofs)}
    for future in as_completed(futures):
        results.append(future.result())
    return sorted(results, key=lambda x: x["idx"])

def process_problem(prob: dict, prob_idx: int, args, executor: ThreadPoolExecutor) -> dict:
    """Process a single problem, return result dict."""
    prob_id = prob["id"]

    log(f"  Generating {args.k} candidates...", end=" ")
    proofs = generate_proofs(prob["theorem"], args.k)
    log(f"got {len(proofs)}")

    if not proofs:
        log(f"  ✗ No proofs generated")
        return {"id": prob_id, "success": False, "phase": "propose", "attempts": [], "pass_at_1": False, "repair_win": False}

    # Verify in parallel
    log(f"  Verifying...", end=" ")
    verify_results = verify_many(prob["header"], prob["theorem"], proofs, args.max_parallel, executor, prob_idx)

    # Check for success
    successes = [r for r in verify_results if r["success"]]
    if successes:
        winner = successes[0]
        log(f"✓ SOLVED (candidate {winner['idx']}, {winner['time_ms']}ms)")
        return {
            "id": prob_id, "success": True, "phase": "propose",
            "winning_idx": winner["idx"], "proof": winner["proof"],
            "attempts": verify_results,
            "pass_at_1": winner["idx"] == 0,
            "repair_win": False
        }

    # All failed - try repair
    log(f"all {len(proofs)} failed")

    if args.no_repair:
        return {"id": prob_id, "success": False, "phase": "propose", "attempts": verify_results, "pass_at_1": False, "repair_win": False}

    # Pick best failure (prefer shorter errors)
    best_failure = min(verify_results, key=lambda x: len(x.get("error_excerpt", "") or ""))

    log(f"  Repairing (based on candidate {best_failure['idx']})...", end=" ")
    repairs = generate_repairs(prob["theorem"], best_failure["proof"], best_failure["error_excerpt"] or "", args.r)
    log(f"got {len(repairs)}")

    repair_results = []
    if repairs:
        log(f"  Verifying repairs...", end=" ")
        repair_results = verify_many(prob["header"], prob["theorem"], repairs, args.max_parallel, executor, prob_idx + 1000)
        repair_successes_list = [r for r in repair_results if r["success"]]

        if repair_successes_list:
            winner = repair_successes_list[0]
            log(f"✓ SOLVED via repair ({winner['time_ms']}ms)")
            return {
                "id": prob_id, "success": True, "phase": "repair",
                "proof": winner["proof"],
                "propose_attempts": verify_results,
                "repair_attempts": repair_results,
                "pass_at_1": False,
                "repair_win": True
            }
        log(f"all repairs failed")

    log(f"  ✗ FAILED")
    return {
        "id": prob_id, "success": False, "phase": "repair" if repairs else "propose",
        "propose_attempts": verify_results,
        "repair_attempts": repair_results,
        "pass_at_1": False,
        "repair_win": False
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="Number of candidates to generate")
    parser.add_argument("--r", type=int, default=2, help="Number of repair candidates")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel Lean verifications")
    parser.add_argument("--no-repair", action="store_true", help="Disable repair phase")
    args = parser.parse_args()

    # Setup
    global client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY")
        return
    client = genai.Client(api_key=api_key)

    problems_path = Path(__file__).parent / "problems" / "problems.json"
    with open(problems_path) as f:
        problems = json.load(f)

    results = []
    solved = 0
    pass_at_1 = 0
    repair_successes = 0

    print(f"V1 Benchmark: K={args.k}, R={args.r}, max_parallel={args.max_parallel}")
    print(f"Running on {len(problems)} problems...\n")

    # Shared executor for all Lean verifications
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        for i, prob in enumerate(problems):
            log(f"[{i+1}/{len(problems)}] {prob['id']}")

            result = process_problem(prob, i, args, executor)
            results.append(result)

            if result["success"]:
                solved += 1
            if result.get("pass_at_1"):
                pass_at_1 += 1
            if result.get("repair_win"):
                repair_successes += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"SCORE: {solved}/{len(problems)}")
    print(f"pass@1: {pass_at_1}/{len(problems)}")
    print(f"pass@{args.k}: {solved - repair_successes}/{len(problems)}")
    print(f"repair successes: {repair_successes}")
    print(f"{'='*50}")

    # Save
    results_path = Path(__file__).parent / "results" / f"v1_k{args.k}_r{args.r}_{int(time.time())}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "config": {"k": args.k, "r": args.r, "max_parallel": args.max_parallel},
            "score": f"{solved}/{len(problems)}",
            "pass_at_1": pass_at_1,
            "repair_successes": repair_successes,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
