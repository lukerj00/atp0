"""Shared utilities for ATP benchmarks."""

import hashlib
import json
import os
import time
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file from module directory
load_dotenv(Path(__file__).parent / ".env")

# ============== Configuration ==============

LEAN_PROJECT = Path(__file__).parent / "lean_project"
KIMINA_URL = "https://lean.cajal.org/api/check"
TIMEOUT = 120
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ============== Prompts ==============

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

REPAIR_PROMPT = """You are repairing a Lean 4 proof.

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

# ============== Logging ==============

_print_lock = Lock()

def log(msg: str, end="\n", flush=True):
    """Thread-safe print."""
    with _print_lock:
        print(msg, end=end, flush=flush)

# ============== Client Setup ==============

def create_client() -> genai.Client:
    """Create Gemini client with proper timeout and retry config."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    # Set both to avoid SDK warning
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

    return genai.Client(
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

ERROR_PRIORITY = {
    "unsolved_goals": 0,
    "tactic_failed": 1,
    "type_mismatch": 2,
    "unknown_identifier": 3,
    "parse_error": 4,
    "sorry": 5,
    "other": 6
}

# ============== Proof Utilities ==============

def clean_proof(proof: str) -> str:
    """Clean proof text: remove markdown, ensure starts with 'by'."""
    if "```" in proof:
        proof = "\n".join(l for l in proof.split("\n") if "```" not in l)
    proof = proof.strip()
    if not proof.startswith("by"):
        proof = f"by\n  {proof}"
    return proof

def normalize_proof(proof: str) -> str:
    """Normalize proof for deduplication."""
    lines = [line.strip() for line in proof.strip().split("\n") if line.strip()]
    return "\n".join(lines)

def proof_hash(proof: str) -> str:
    """Hash normalized proof for caching."""
    return hashlib.md5(normalize_proof(proof).encode()).hexdigest()[:12]

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

def generate_proofs(client: genai.Client, theorem: str, k: int) -> list[str]:
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

def generate_repairs(client: genai.Client, theorem: str, failed_proof: str,
                     error: str, error_class: str, r: int) -> list[str]:
    """Generate R repair candidates with error-class-aware prompt."""
    prompt = REPAIR_PROMPT.format(
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

def verify_batch_server(header: str, theorem: str, proofs: list[str],
                        timeout: int = 60) -> list[dict]:
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
