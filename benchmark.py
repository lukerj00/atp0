#!/usr/bin/env python3
"""Minimal single-shot Lean 4 prover benchmark using Gemini Flash."""

import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file from script directory
load_dotenv(Path(__file__).parent / ".env")

MODEL = "gemini-2.5-flash"
TIMEOUT = 120

SYSTEM_PROMPT = """You are a Lean 4 theorem prover. Given a theorem statement, produce a complete proof.

RULES:
1. Output ONLY the proof tactics (starting with `by`) - no explanation, no markdown
2. Use Mathlib tactics: simp, ring, linarith, nlinarith, omega, norm_num, field_simp, polyrith, aesop, decide, exact, apply, intro, constructor, cases, induction, rfl, ext, funext, use, obtain, have, calc, rw, conv
3. Keep proofs concise - prefer powerful automation
4. For multiple goals use `·` bullets or `<;>` combinator
5. Output nothing except the proof starting with `by`"""

client = None

def generate_proof(statement: str, retries: int = 3) -> str:
    """Call Gemini to generate a proof with retry on rate limit."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=f"Prove this theorem:\n\n{statement}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(15 * (attempt + 1))  # backoff
                continue
            raise


LEAN_PROJECT = Path(__file__).parent / "lean_project"

def verify_lean(header: str, theorem: str, proof: str) -> tuple[bool, str]:
    """Verify a Lean proof by compiling it in the pre-built project."""
    # Clean proof - remove markdown fences if present
    if "```" in proof:
        lines = proof.split("\n")
        proof = "\n".join(l for l in lines if "```" not in l)
    proof = proof.strip()
    if not proof.startswith("by"):
        proof = "by\n  " + proof

    # Write proof to the existing project
    lean_code = f'''{header}

{theorem} :=
{proof}
'''
    (LEAN_PROJECT / "Verify.lean").write_text(lean_code)

    # Run lake build
    try:
        result = subprocess.run(
            ["lake", "build", "Verify"],
            cwd=LEAN_PROJECT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def main():
    # Load problems
    problems_path = Path(__file__).parent / "problems" / "problems.json"
    with open(problems_path) as f:
        problems = json.load(f)

    # Setup Gemini
    global client
    # Prefer GEMINI_API_KEY, fall back to GOOGLE_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
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

    results = []
    solved = 0

    print(f"Running benchmark on {len(problems)} problems...\n")

    for i, prob in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {prob['id']}: ", end="", flush=True)

        # Generate proof
        try:
            proof = generate_proof(prob["theorem"])
        except Exception as e:
            print(f"GENERATION ERROR: {e}")
            results.append({"id": prob["id"], "success": False, "error": str(e)})
            continue

        # Verify
        success, output = verify_lean(prob["header"], prob["theorem"], proof)

        if success:
            print("✓ SOLVED")
            solved += 1
        else:
            error_preview = output[:100].replace("\n", " ") if output else "unknown"
            print(f"✗ FAILED ({error_preview}...)")

        results.append({
            "id": prob["id"],
            "description": prob.get("description", ""),
            "theorem": prob["theorem"],
            "success": success,
            "proof": proof,
            "output": output[:500] if output else None
        })
        time.sleep(2)  # rate limit between requests

    # Summary
    print(f"\n{'='*50}")
    print(f"SCORE: {solved}/{len(problems)}")
    print(f"{'='*50}")

    # Save results
    results_path = Path(__file__).parent / "results" / f"v0_{int(time.time())}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "config": {"version": "v0", "model": MODEL},
            "score": f"{solved}/{len(problems)}",
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
