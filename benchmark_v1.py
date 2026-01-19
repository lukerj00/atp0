#!/usr/bin/env python3
"""V1 benchmark: pass@K with strategy diversity + 1-step repair."""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from google.genai import types

from common import (
    LEAN_PROJECT, MODEL, TIMEOUT,
    PROPOSE_PROMPT,
    create_client, log,
    clean_proof, generate_proofs,
    verify_batch_server
)

# V1 uses simpler repair prompt without error classification
REPAIR_PROMPT_V1 = """You are repairing Lean 4 proofs.

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

# Module-level client (set in main)
client = None

def generate_repairs_v1(theorem: str, failed_proof: str, error: str, r: int) -> list[str]:
    """Generate R repair candidates (V1 simple prompt)."""
    prompt = REPAIR_PROMPT_V1.format(theorem=theorem, failed_proof=failed_proof, error=error[:500], r=r)
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
                time.sleep(15 * (attempt + 1))
                continue
            return []
    return []

# ============== Local Verification ==============

def verify_proof_local(header: str, theorem: str, proof: str, idx: int, prob_idx: int = 0) -> dict:
    """Verify a single proof locally, return result dict."""
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
            "error_excerpt": error_excerpt,
            "output": output[:500]
        }
    except subprocess.TimeoutExpired:
        return {"idx": idx, "success": False, "time_ms": TIMEOUT*1000, "proof": proof, "error_excerpt": "TIMEOUT", "output": "TIMEOUT"}
    except Exception as e:
        return {"idx": idx, "success": False, "time_ms": 0, "proof": proof, "error_excerpt": str(e), "output": str(e)}
    finally:
        verify_file.unlink(missing_ok=True)

def verify_many_local(header: str, theorem: str, proofs: list[str], executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
    """Verify multiple proofs in parallel using local Lean."""
    results = []
    futures = {executor.submit(verify_proof_local, header, theorem, p, i, prob_idx): i for i, p in enumerate(proofs)}
    for future in as_completed(futures):
        results.append(future.result())
    return sorted(results, key=lambda x: x["idx"])

def verify_proofs(header: str, theorem: str, proofs: list[str], args, executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
    """Verify proofs using server (default) or local Lean."""
    if args.local:
        return verify_many_local(header, theorem, proofs, executor, prob_idx)
    else:
        results = verify_batch_server(header, theorem, proofs, timeout=args.timeout)
        return sorted(results, key=lambda x: x["idx"])

# ============== Problem Processing ==============

def process_problem(prob: dict, prob_idx: int, args, executor: ThreadPoolExecutor) -> dict:
    """Process a single problem, return result dict."""
    prob_id = prob["id"]

    log(f"  Generating {args.k} candidates...", end=" ")
    proofs = generate_proofs(client, prob["theorem"], args.k)
    log(f"got {len(proofs)}")

    if not proofs:
        log(f"  ✗ No proofs generated")
        return {"id": prob_id, "success": False, "phase": "propose", "attempts": [], "pass_at_1": False, "repair_win": False}

    # Verify
    log(f"  Verifying...", end=" ")
    verify_results = verify_proofs(prob["header"], prob["theorem"], proofs, args, executor, prob_idx)

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
    repairs = generate_repairs_v1(prob["theorem"], best_failure["proof"], best_failure["error_excerpt"] or "", args.r)
    log(f"got {len(repairs)}")

    repair_results = []
    if repairs:
        log(f"  Verifying repairs...", end=" ")
        repair_results = verify_proofs(prob["header"], prob["theorem"], repairs, args, executor, prob_idx + 1000)
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

# ============== Main ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="Number of candidates to generate")
    parser.add_argument("--r", type=int, default=2, help="Number of repair candidates")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel local Lean verifications")
    parser.add_argument("--no-repair", action="store_true", help="Disable repair phase")
    parser.add_argument("--problems", type=str, default="problems.json", help="Problem file to use (in problems/)")
    parser.add_argument("--local", action="store_true", help="Use local Lean instead of Kimina server")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per verification batch (server mode)")
    args = parser.parse_args()

    global client
    client = create_client()

    problems_path = Path(__file__).parent / "problems" / args.problems
    with open(problems_path) as f:
        problems = json.load(f)

    results = []
    solved = 0
    pass_at_1 = 0
    repair_successes = 0

    mode = "local" if args.local else "server"
    print(f"V1 Benchmark: K={args.k}, R={args.r}, mode={mode}, model={MODEL}, problems={args.problems}")
    print(f"Running on {len(problems)} problems...\n")

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
    prob_tag = args.problems.replace(".json", "").replace("problems_", "").replace("problems", "default")
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    timestamp_readable = now.strftime("%y%m%d-%H%M")
    results_path = Path(__file__).parent / "results" / f"v1_{prob_tag}_k{args.k}_r{args.r}_{timestamp_readable}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S PST"),
            "config": {"version": "v1", "model": MODEL, "k": args.k, "r": args.r, "mode": mode, "timeout": args.timeout, "max_parallel": args.max_parallel, "problems": args.problems},
            "score": f"{solved}/{len(problems)}",
            "pass_at_1": pass_at_1,
            "repair_successes": repair_successes,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
