#!/usr/bin/env python3
"""V1 benchmark: pass@K with strategy diversity + 1-step repair."""

import argparse
import json
import os
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
MODEL = "gemini-2.5-flash"

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
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
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
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
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

def clean_proof(proof: str) -> str:
    """Clean proof text: remove markdown, ensure starts with 'by'."""
    if "```" in proof:
        proof = "\n".join(l for l in proof.split("\n") if "```" not in l)
    proof = proof.strip()
    if not proof.startswith("by"):
        proof = f"by\n  {proof}"
    return proof

def verify_batch_server(header: str, theorem: str, proofs: list[str], timeout: int = 60) -> list[dict]:
    """Verify multiple proofs in a single batch request to Kimina server."""
    # Build snippets for batch request
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
            # Check for top-level error
            error = res.get("error")
            resp = res.get("response") or {}

            # Check for error messages in response
            messages = resp.get("messages", [])
            error_msgs = [m for m in messages if m.get("severity") == "error"]

            # Check for sorry
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
                "error_excerpt": error_excerpt,
                "output": str(res)[:500]
            })
        return sorted(results, key=lambda x: x["idx"])
    except Exception as e:
        # Return all failures on error
        return [
            {"idx": i, "success": False, "time_ms": 0, "proof": clean_proof(p),
             "error_excerpt": f"Server error: {e}", "output": str(e)}
            for i, p in enumerate(proofs)
        ]

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

def verify_many_local(header: str, theorem: str, proofs: list[str], max_parallel: int, executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
    """Verify multiple proofs in parallel using local Lean (slow)."""
    results = []
    futures = {executor.submit(verify_proof_local, header, theorem, p, i, prob_idx): i for i, p in enumerate(proofs)}
    for future in as_completed(futures):
        results.append(future.result())
    return sorted(results, key=lambda x: x["idx"])

def verify_proofs(header: str, theorem: str, proofs: list[str], args, executor: ThreadPoolExecutor, prob_idx: int = 0) -> list[dict]:
    """Verify proofs using server (default) or local Lean."""
    if args.local:
        return verify_many_local(header, theorem, proofs, args.max_parallel, executor, prob_idx)
    else:
        return verify_batch_server(header, theorem, proofs, timeout=args.timeout)

def process_problem(prob: dict, prob_idx: int, args, executor: ThreadPoolExecutor) -> dict:
    """Process a single problem, return result dict."""
    prob_id = prob["id"]

    log(f"  Generating {args.k} candidates...", end=" ")
    proofs = generate_proofs(prob["theorem"], args.k)
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
    repairs = generate_repairs(prob["theorem"], best_failure["proof"], best_failure["error_excerpt"] or "", args.r)
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

    # Setup
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
    repair_successes = 0

    mode = "local" if args.local else "server"
    print(f"V1 Benchmark: K={args.k}, R={args.r}, mode={mode}, problems={args.problems}")
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
