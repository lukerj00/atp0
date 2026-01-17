# Milestone 6 — Production Hardening + GCP Deployment (LeanWorker Pool + Orchestrator + Retriever)

## Goal
Take the Milestone 1–5 local system and make it **production-deployable on GCP** with:
- reliable container builds
- hardened runtime limits / isolation
- scalable LeanWorker pool (the expensive part)
- observable and debuggable orchestration (metrics/traces/logging)
- persistent artifact + trace storage
- CI/CD + reproducible infrastructure

This milestone is about **systems engineering**, not improving proof intelligence.

---

## Dependencies
- Milestone 1: LeanWorker (batch) + Milestone 5 optional stepwise API
- Milestone 2–4: Orchestrator + Model Gateway + Planner + Retriever integration
- Milestone 3: Retriever service
- Milestone 5: Stepwise mode + budgets (strongly recommended before production)

---

## Scope (what to build)

### Deliverables
1. Containerization + pinned builds for:
   - LeanWorker
   - Orchestrator
   - Retriever
2. GCP infrastructure:
   - Artifact Registry
   - Cloud Run services (Orchestrator, Retriever, optional Gateway)
   - LeanWorker pool (Cloud Run or GKE — choose based on throughput)
   - Cloud Storage bucket for artifacts/traces
   - Firestore (or Cloud SQL) for metadata + cache
   - Cloud Logging/Monitoring dashboards
3. Service-to-service authentication and network hardening
4. Operational limits:
   - request size limits
   - concurrency limits
   - timeouts
   - CPU/memory caps
5. CI pipeline:
   - build images
   - push to Artifact Registry
   - run smoke tests
   - deploy to staging
6. Benchmark runner job:
   - periodic eval on a fixed set
   - produce regression reports

---

## Architecture on GCP (recommended V0)

### Services
1. **Orchestrator (Cloud Run)**
   - handles `/prove` API
   - schedules proof attempts to LeanWorker
   - calls Retriever and Model Gateway backends
   - stores traces to GCS + metadata to Firestore

2. **Retriever (Cloud Run)**
   - serves `/search`
   - reads SQLite index packaged in image (read-only)

3. **LeanWorker Pool**
   Choose one:
   - **Option A: Cloud Run** (simplest ops, good for moderate throughput)
   - **Option B: GKE** (best for heavy throughput + persistent warm workers)

**Recommended: start with Cloud Run for V0.** Move to GKE only if:
- compilation times are too high,
- concurrency needs are high,
- you need persistent Lean sessions across requests reliably.

4. **(Optional) Model Gateway**
   - If using external model APIs, Orchestrator can call them directly
   - If you want unified accounting & retries, deploy Gateway as Cloud Run

### Storage
- **GCS**: proof artifacts + trace logs + benchmark reports
- **Firestore**: job metadata, run summaries, caches (later: Redis for hot cache)

---

## Infrastructure as Code (IaC)

### Required deliverables
- `infra/terraform/` (preferred) OR `infra/gcloud/` scripts
- must create:
  - Artifact Registry repo
  - Cloud Run services
  - GCS bucket + lifecycle rules
  - Firestore database (native mode)
  - IAM service accounts and bindings
  - (optional) Pub/Sub topics if you queue tasks

### Service accounts (least privilege)
- `sa-orchestrator`:
  - write to GCS bucket
  - read/write Firestore
  - invoke LeanWorker and Retriever services
- `sa-leanworker`:
  - no GCS write needed (return artifacts to orchestrator)
  - minimal permissions
- `sa-retriever`:
  - minimal permissions

---

## Container hardening

### LeanWorker hardening (critical)
- Enforce:
  - CPU/memory limits at container level
  - strict per-request timeouts (already in code)
  - max request payload size (to prevent huge decls)
- Disable outbound network egress if possible (VPC connector / firewall)
- Run as non-root user (recommended)
- Limit temp directory usage and cleanup sessions

### Orchestrator hardening
- Rate limiting (Cloud Run + app-level)
- Maximum concurrent proof jobs
- BudgetManager enforced server-side (no client override beyond caps)
- Validate theorem inputs (length, allowed chars)

---

## Observability (must-have)

### Logging (Cloud Logging)
Log structured JSON per attempt with fields:
- job_id, theorem_hash, blueprint_id/version
- session_id (if stepwise)
- candidate_id, round/step
- lean_ok, error_class, time_ms
- model backend, model_name, tokens (if available)
- budget remaining

### Metrics (Cloud Monitoring)
Export:
- requests/sec by endpoint
- lean checks/sec, mean check time
- success rate
- time-to-proof distribution
- error class frequency
- worker saturation (concurrency)
- costs: model calls, lean CPU seconds

### Tracing (optional but recommended)
- OpenTelemetry traces across Orchestrator → LeanWorker calls
- Useful for tail latencies and debugging

---

## Artifact and trace persistence

### Artifact schema
For each proof job:
- `jobs/<job_id>/input.json`
- `jobs/<job_id>/final.lean` (if success)
- `jobs/<job_id>/blueprints.json` (versions)
- `jobs/<job_id>/attempts.jsonl`
- `jobs/<job_id>/summary.json`

### Lifecycle rules
- keep successful proofs longer (e.g., 90 days)
- delete raw attempts after shorter time (e.g., 14 days) to control cost

---

## Deployment details

### Cloud Run settings (recommended defaults)
- Orchestrator:
  - CPU: 2 vCPU
  - memory: 2–4 GB
  - concurrency: 10 (tune)
  - request timeout: 15–60 minutes depending on budgets
- Retriever:
  - CPU: 1 vCPU
  - memory: 1–2 GB
  - concurrency: 40
- LeanWorker:
  - CPU: 2–4 vCPU
  - memory: 4–8 GB
  - concurrency: 1–2 (Lean is CPU-heavy)
  - request timeout: ~5–15 minutes max

### Image hosting
- push all images to Artifact Registry
- tag with git SHA
- orchestrator config references exact image digest

---

## CI/CD pipeline

### Required deliverables
- GitHub Actions (or Cloud Build) pipeline:
  1. lint + unit tests
  2. docker build each service
  3. run smoke tests locally (docker compose)
  4. push images to Artifact Registry
  5. deploy to staging Cloud Run
  6. run staging smoke tests
  7. (manual approval) deploy to prod

### Smoke tests (must pass)
- LeanWorker `/healthz` + `/check` trivial theorem
- Retriever `/search` returns at least one result for a known lemma
- Orchestrator `/prove` succeeds on a trivial case using dummy backend

---

## Benchmark runner (regression testing)

### Design
- A scheduled job (Cloud Run Job or Cloud Scheduler + Cloud Run endpoint)
- Runs a fixed set of benchmark problems daily:
  - records solve rate, time, cost
  - stores report to GCS
  - alerts on regressions (Cloud Monitoring alert)

### Required deliverables
- `benchmarks/runner.py` that can:
  - call Orchestrator `/prove` on each case
  - write results to GCS and Firestore

---

## Security and privacy

### API auth
- Use Cloud Run IAM + identity tokens for internal service calls
- Public `/prove` endpoint:
  - protect behind API Gateway / Identity-Aware Proxy (IAP) or your auth layer
- Ensure logs do not include secrets (model API keys, etc.)

### Secrets management
- Store API keys in Secret Manager
- Inject via Cloud Run environment variables
- Rotate keys periodically

---

## Definition of Done
1. All services build reproducibly and are pushed to Artifact Registry
2. Services deployed on GCP:
   - Orchestrator, Retriever, LeanWorker pool
3. Orchestrator can solve trivial benchmark cases end-to-end on GCP
4. Logs/metrics are visible in Cloud Logging/Monitoring
5. Proof artifacts and traces persist in GCS
6. CI/CD deploys to staging automatically; production deploy gated
7. Basic benchmark runner job executes and writes a report

---

## Notes for Milestone 7+
- Move LeanWorker pool to GKE if throughput requires persistent workers
- Add robust stepwise goal-state extraction with Lean server protocol
- Add stronger caching layer (Redis/Memorystore)
- Add multi-model routing and cost-optimized escalation policies

---