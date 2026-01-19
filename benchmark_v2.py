#!/usr/bin/env python3
"""V2 benchmark: Multi-round verification-guided repair + caching."""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from common import (
    LEAN_PROJECT, MODEL, TIMEOUT,
    create_client, log,
    classify_error, clean_proof, proof_hash, dedupe_proofs,
    generate_proofs, generate_repairs,
    verify_batch_server, select_best_failures
)

# Module-level client (set in main)
client = None

# ============== Local Verification ==============

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

def verify_many_local(header: str, theorem: str, proofs: list[str], executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
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
        return verify_many_local(header, theorem, proofs, executor, prob_idx)
    else:
        return verify_batch_server(header, theorem, proofs, timeout=args.timeout)

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
    proofs = generate_proofs(client, theorem, args.k)
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
    round_num = 0
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

            repairs = generate_repairs(
                client, theorem, failure["proof"],
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
        "rounds_used": round_num + 1,
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
    client = create_client()

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
    print(f"V2 Benchmark: K={args.k}, max_rounds={args.max_rounds}, mode={mode}, model={MODEL}, problems={args.problems}")
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
