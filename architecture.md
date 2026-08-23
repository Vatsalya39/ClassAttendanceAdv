# Phased Build Prompts — Face Recognition Attendance System (Multi-Campus Scale)

Feed these to your coding agent **one phase at a time**, in order. Wait for each
phase to complete and verify it before moving to the next. Each prompt assumes
the outputs of the previous phase exist in the repo.

This is a scale-revised version of the original single-classroom plan. Read
this note before you start — it explains *why* each phase differs from the
60–70-student version, so you (and the agent) don't accidentally re-introduce
assumptions that only work at classroom scale.

## What changes at 20,000–50,000 students, and why

- **"Cluster" = organizational hierarchy, not a database shard.** A section
  (60–70 students) belongs to a department, which belongs to an institution.
  Face matching for any one class session is *always* scoped to that
  section's roster — never a global search across all students. This is the
  single most important design rule in this whole system: it's what keeps
  match latency identical whether you have 5,000 or 50,000 students enrolled.
  pgvector's HNSW index can comfortably hold low millions of 512-d vectors on
  one node, so you do **not** need to shard embeddings across databases at
  this scale — you need scoped queries.
- **Attendance capture becomes async.** At classroom scale, one admin
  uploading one photo and waiting 2 seconds for a response is fine. At campus
  scale, dozens or hundreds of sections may all capture within the same
  5-minute window at the start of a period. The capture endpoint enqueues a
  job and returns immediately; a worker pool (scaled independently) does the
  detection/embedding/matching and writes results.
- **The attendance table needs partitioning.** 20,000 students × ~200
  session-days/year is ~4M rows/year; at 50,000 students that's ~10M/year.
  Un-partitioned, this table degrades reporting queries and bloats vacuum
  time. Partition by range on `session_date` (monthly).
- **GPU inference scales with queue depth, not with a fixed replica count.**
  Use KEDA-style scaling rules on your message queue so the GPU tier scales
  to zero when idle (cost) and scales out fast when many sections capture at
  once (latency).
- **Admin roles need a hierarchy too.** A single flat "admin" role doesn't
  work when you have per-department or per-institution administrators. Add
  role scoping (super_admin / institution_admin / department_admin /
  faculty) so a department admin can't see or resolve another department's
  review queue.

---

Context to paste once at the start of the conversation with the agent:

```
We are building a production face-recognition attendance system operating at
multi-campus scale: up to 50,000 enrolled students, organized into sections
of 60-70 students each (a section is the unit a class session is taken
against). Many sections may run attendance capture concurrently.

Stack: FastAPI backend, React/Next.js frontend, PostgreSQL (Azure Database
for PostgreSQL Flexible Server) with pgvector extension, Azure Blob Storage
for images, Azure Service Bus for the capture job queue, Azure Cache for
Redis (roster caching, rate limiting, session data), Azure Container Apps
for hosting (CPU profile for API/workers, GPU workload profile for
inference, KEDA-scaled on queue depth), Azure Static Web Apps for frontend,
Azure Key Vault for secrets.

Face pipeline: RetinaFace (via InsightFace buffalo_l) for detection, 5-point
alignment, ArcFace 512-d embeddings, cosine similarity matching via pgvector
(hnsw index), 3-5 stored embeddings per student, no forced matches below a
calibrated threshold — unmatched faces go to an admin review queue instead
of being auto-marked absent/present.

Hard architectural rule: face matching for a class session is ALWAYS scoped
to that section's enrolled roster (a join, ~60-70 candidate embeddings),
never a global search across all enrolled students. This must hold
regardless of total student count. Review-queue manual search (an admin
looking up "who is this unmatched face") is the one legitimate case for a
broader, institution-scoped search, and it's expected to be slower and
used rarely.

Attendance capture is asynchronous: the capture endpoint enqueues a job and
returns a job id; a separately-scaled worker pool processes it.

We will build this in phases. Confirm you understand before we start Phase 0.
```

---

## Phase 0 — Repo & environment scaffold

```
Set up the project repository structure for this system:
- /backend (FastAPI app, Python 3.11+) — split into api (HTTP layer) and
  worker (queue consumer) entry points sharing the same core package
- /frontend (React + Vite or Next.js, your call — explain which and why)
- /infra (empty for now, will hold IaC)
- /ml (face pipeline module, isolated from the API/worker layer)
- /docs

Set up:
- pyproject.toml or requirements.txt for backend with FastAPI, uvicorn,
  sqlalchemy, alembic, psycopg (or asyncpg), passlib[argon2], python-jose
  (JWT), pydantic-settings, redis client, azure-servicebus (or a pluggable
  queue interface so local dev can use a simple in-process/Redis-backed
  queue instead of real Azure Service Bus)
- Pre-commit hooks (black, ruff, isort) for backend
- .env.example listing every required environment variable (DB connection,
  JWT secret, Azure Blob connection string, Redis connection string, Service
  Bus/queue connection string, Key Vault URI) — do NOT put real secrets
  anywhere
- README with local dev setup instructions
- A docker-compose.yml for local dev: postgres (with pgvector image), redis,
  a local queue (or Service Bus emulator if you use one), backend-api,
  backend-worker, frontend

Do not implement any business logic yet — this phase is scaffolding only.
```

---

## Phase 1 — Database layer (with organizational hierarchy + partitioning)

```
Implement the database layer using SQLAlchemy models + Alembic migrations for
this schema (adapt as needed but keep the intent):

Organizational hierarchy:
- institutions: institution_id (uuid pk), name, created_at
- departments: department_id (uuid pk), institution_id (fk), name
- sections: section_id (uuid pk), department_id (fk), name (e.g. "CSE-3B"),
  academic_year, created_at — this is the ~60-70-student cluster
- enrollments: enrollment_id (uuid pk), student_id (fk), section_id (fk),
  UNIQUE(student_id, section_id)

Core entities:
- students: student_id (uuid pk), roll_number (unique), name, password_hash,
  institution_id (fk), created_at
- face_embeddings: embedding_id (uuid pk), student_id (fk), embedding
  (pgvector, 512 dims), source_image_url, created_at
- class_sessions: session_id (uuid pk), section_id (fk), course_name,
  session_date, admin_id (fk, who ran capture), created_at
- attendance: attendance_id (uuid pk), student_id (fk), session_id (fk),
  session_date, match_distance, marked_at, UNIQUE(student_id, session_id) —
  PARTITION this table BY RANGE on session_date, monthly partitions, with a
  helper (or pg_partman) to create new partitions ahead of time
- admins: admin_id (uuid pk), username (unique), password_hash, role
  (super_admin / institution_admin / department_admin / faculty),
  institution_id (fk, nullable for super_admin), department_id (fk, nullable)
- review_queue: review_id (uuid pk), session_id (fk), face_crop_url,
  best_guess_student_id (nullable), best_guess_distance (nullable), status
  (pending/resolved), created_at
- audit_log: log_id (uuid pk), admin_id (fk), action, target_type, target_id,
  created_at

Write the Alembic migration that also runs `CREATE EXTENSION IF NOT EXISTS
vector;` and creates an hnsw index on face_embeddings.embedding using
vector_cosine_ops. Document explicitly in a comment: this single hnsw index
is sufficient at 50k+ students because it is never queried unfiltered — see
find_nearest_embeddings_for_roster below.

Add a repository/DAO layer with clear functions, and make the scoping
explicit in the function names:
- create_student, add_embedding, mark_attendance, get_attendance_for_date,
  enqueue_review, resolve_review
- find_nearest_embeddings_for_roster(vector, section_id, top_k) — joins
  face_embeddings to enrollments filtered by section_id BEFORE the vector
  search, so the ANN search only ever ranks against that section's ~60-70
  candidates, not the whole table
- find_nearest_embeddings_institution_wide(vector, institution_id, top_k) —
  the deliberately slower, explicitly-named fallback used only by the manual
  review-queue "search all students" admin action
- create_institution, create_department, create_section, bulk_enroll
  (accepts a list of student_ids + a section_id, for CSV import in Phase 4)

Write unit tests against a test database confirming: (a) roster-scoped
search only ever considers embeddings belonging to students enrolled in the
given section, even when other sections' embeddings exist in the same table,
and (b) attendance partition pruning works — a query filtered on
session_date only scans the relevant partition (check via EXPLAIN).
```

---

## Phase 2 — Face pipeline module (standalone, testable, queue-worker ready)

```
Build the /ml face pipeline module, fully decoupled from the API/worker
layer, with these functions:
- detect_faces(image) -> list of bounding boxes + 5-point landmarks
  (InsightFace RetinaFace)
- align_and_embed(image, landmarks) -> 512-d ArcFace embedding
- batch_process(image) -> list of {bbox, embedding} for every face in one
  classroom photo
- match_embedding(query_vector, candidates, threshold) -> best match or None
  (candidates is always a pre-filtered roster-sized list — this function has
  no knowledge of the total student population and should never receive
  more than a few hundred candidates)

Design the module so the model is loaded once per process (warm start) and
reused across many batch_process calls, since it will run inside a
long-lived queue worker, not a fresh process per request.

Write a CLI script (scripts/test_pipeline.py) for a single image path.

Write scripts/calibrate_threshold.py that takes a folder of labeled
known-same/known-different pairs and outputs the distance distribution +
suggested threshold.

Write scripts/bench_throughput.py that measures faces/second on this
machine's GPU for batch sizes matching realistic classroom photos, and
estimates how many concurrent worker replicas are needed to keep queue wait
time under a target (e.g. 30s) if N sections capture within the same minute
— this feeds directly into the KEDA scale-rule numbers in Phase 6.

Do not call any database, queue, or API code from this module.
```

---

## Phase 3 — Backend API + async worker

```
Build the FastAPI backend and a separate worker process using the DAO layer
from Phase 1 and the pipeline module from Phase 2.

API endpoints:
- POST /register — roll_number, name, password, section_id, 3-5 enrollment
  images -> hash password (argon2), run pipeline on each image, store
  embeddings, enroll into section, return student_id
- POST /auth/student/login — roll_number, password -> JWT
- GET /attendance/me — (student JWT) -> that student's attendance history
- POST /auth/admin/login — username, password -> JWT (role + scope claims:
  institution_id/department_id where applicable)
- POST /admin/session/capture — (admin JWT, must have access to the target
  section) upload one classroom photo + section_id -> validate, store raw
  image, ENQUEUE a capture job {job_id, session_id, image_url}, return
  {job_id, status: "queued"} immediately. Do NOT run the pipeline inline.
- GET /admin/session/capture/{job_id} — poll job status ->
  {status: queued|processing|done|failed, matched: [...], review_needed: [...]}
  when done
- GET /admin/review-queue?section_id= — (admin JWT, scoped) list pending
  review items for sections the admin can access
- POST /admin/review-queue/{id}/resolve — assign to a student (roster search
  defaults to the session's section; falls back to
  find_nearest_embeddings_institution_wide only on explicit "search all
  students" action) or discard
- GET /admin/attendance?section_id=&date= — attendance report
- Institution/department/section CRUD (super_admin / scoped admin only) +
  POST /admin/sections/{id}/bulk-enroll (CSV of roll numbers)

Worker (backend/worker entry point):
- Consumes capture jobs from the queue
- Runs batch_process on the image, then for each detected face calls
  find_nearest_embeddings_for_roster scoped to the job's section_id
- Writes attendance or review_queue rows, updates job status
- Idempotent: re-processing the same job_id must not double-write attendance
  (respect the UNIQUE(student_id, session_id) constraint and handle the
  conflict cleanly)

Add JWT auth middleware enforcing role + institution/department scope on
every admin route (a department_admin for Dept A must get 403 touching Dept
B's sections). Add per-admin and per-section rate limiting on
/admin/session/capture using Redis, so one runaway client can't flood the
queue. Use the Storage protocol from before so local filesystem and Azure
Blob are swappable. Write integration tests for each endpoint, including a
test that submits captures for two different sections concurrently and
asserts each only matches against its own roster.
```

---

## Phase 4 — Frontend

```
Build the React frontend with:

- Home page: Register / Check Attendance / Admin Login
- Registration flow: name/roll number/password/section selection, then
  guided camera capture (3-5 shots), submit to POST /register
- Student login + attendance view
- Admin login -> JWT (httpOnly cookie preferred)
- Admin dashboard, scoped to the admin's institution/department:
  - Section picker, capture/upload classroom photo
  - Poll GET /admin/session/capture/{job_id} with a progress indicator
    (queued -> processing -> done) instead of blocking on the request
  - Show matched vs needs-review results once done
  - Review queue UI to resolve unmatched faces (default search scoped to
    the session's section roster; an explicit "search all students" option
    for edge cases)
  - Attendance report view with section + date filters
  - Bulk-enroll UI: upload a CSV of roll numbers against a section
- Super-admin console: create/manage institutions, departments, sections,
  and department_admin accounts

Keep API calls in a single api client module. Handle camera permission
denial gracefully; camera access only over HTTPS/localhost.
```

---

## Phase 5 — Security hardening

```
Review and harden the whole system:
- Confirm all passwords use argon2id with appropriate work factors, never
  logged or returned in any response
- JWT: short-lived access tokens + refresh token flow; role and scope claims
  (institution_id/department_id) are verified server-side on every admin
  route, not trusted from the client
- Rate limiting on /auth/*, /register, and /admin/session/capture
- Input validation and file-type/size limits on all image uploads
- CORS configured to only allow the deployed frontend origin
- Confirm tenant isolation: write a test suite that, for every admin-scoped
  endpoint, asserts a token scoped to institution/department A cannot read
  or write data belonging to institution/department B
- Audit log every admin action (capture triggered, review resolved,
  section/enrollment changes, bulk imports) with admin_id and scope
- Document data handling: face embeddings and images are biometric data —
  add a data retention policy note and a way to delete a student's stored
  images/embeddings on request, including from any partitioned attendance
  history
```

---

## Phase 6 — Azure infrastructure as code (scale-tuned)

```
Write Bicep (or Terraform, your call — explain which and why) for:
- Azure Database for PostgreSQL Flexible Server with pgvector enabled, sized
  for the target load (state your vCore/memory choice and why), plus a
  read replica for reporting/analytics queries so they don't contend with
  live attendance writes
- Azure Cache for Redis (roster/session caching, rate limiting)
- Azure Service Bus namespace + queue for capture jobs, with dead-letter
  handling for failed jobs
- Azure Container Apps environment:
  - api tier (CPU profile, public via ingress, autoscale on HTTP concurrency)
  - worker tier (CPU profile, consumes the queue, scales via KEDA on queue
    message count — explain the scale rule you chose and how it relates to
    the throughput numbers from Phase 2's bench_throughput.py)
  - ml-inference tier (GPU workload profile, internal ingress only, also
    KEDA-scaled on queue depth, min replicas 0 to control cost during idle
    periods, max replicas sized for your worst-case concurrent-sections
    estimate)
- Azure Static Web Apps for the frontend
- Azure Blob Storage for enrollment/session images
- Azure Key Vault for DB connection string, JWT secret, storage and Service
  Bus connection strings — Container Apps pull secrets via managed identity
- Networking: Postgres not publicly reachable, VNet integration between
  Container Apps and the database and Redis
- A partition-maintenance job (Container Apps job on a schedule, or
  pg_partman) that pre-creates the next few months of attendance partitions

Swap the Storage interface implementation from Phase 3 to use Azure Blob SDK.
Output a deployment README with the exact az cli / azd commands to provision
everything from a clean subscription, and a short note on how to raise the
worker/ml-inference max-replica ceilings if enrolled institutions grow
beyond the initial 50,000-student estimate.
```

---

## Phase 7 — CI/CD

```
Set up GitHub Actions (or Azure DevOps, your call) for:
- Backend (api + worker images): lint, type-check, run tests, build
  container images, push to Azure Container Registry, deploy both to
  Container Apps on merge to main
- ml-inference image: separate build/push pipeline (larger image, GPU base)
- Frontend: lint, build, deploy to Static Web Apps on merge to main
- Run Alembic migrations automatically as a pre-deploy step, with explicit
  handling for partitioned-table migrations (creating new partitions is
  never a blocking/locking operation on existing ones) and a manual approval
  gate for production
- Separate workflows/environments for staging vs production
```

---

## Phase 8 — Accuracy calibration & campus-scale load testing

```
Using scripts/calibrate_threshold.py from Phase 2, and a labeled set of real
classroom-condition photos:
- Compute false-accept and false-reject rate at several thresholds, output
  as a table/chart, and recommend a threshold prioritizing precision —
  document the review-queue-volume tradeoff

Scale-specific tests:
- Seed the database with synthetic data at three sizes — 5,000, 20,000, and
  50,000 students across realistic institution/department/section
  hierarchies — and confirm via EXPLAIN ANALYZE that
  find_nearest_embeddings_for_roster's query time is statistically flat
  across all three sizes (this is the proof the roster-scoping design works)
- Simulate a "start of period" burst: N sections (parameterize this, test at
  least 50 and 200) submitting POST /admin/session/capture within the same
  60-second window. Report: queue depth over time, time-to-first-processing,
  end-to-end p50/p95/p99 latency per job, and how fast the ml-inference tier
  scales out to absorb the burst
- Report database connection pool saturation and Redis load under the same
  burst
- Load test /admin/session/capture's synchronous portion (upload + enqueue
  only, since processing is async) for its own response time under
  concurrent load, separately from the end-to-end job latency above
```

---

## Notes for you (not for the agent)

- Do Phases 0–3 with plain local Postgres/Redis/a simple local queue — don't
  provision Azure Service Bus or GPU Container Apps while you're iterating on
  business logic.
- Get real classroom-condition photos early; Phase 8's threshold calibration
  is only as good as the data you calibrate on.
- Treat "review queue rate" as a first-class metric alongside accuracy.
- The single design decision that makes this scale from 60 students to
  50,000 is roster-scoped matching (Phase 1/3). If at any point the agent
  suggests a "search all embeddings" shortcut for convenience, reject it —
  that's the change that would silently reintroduce classroom-scale
  assumptions into a campus-scale system.
- Before committing to specific Postgres SKU sizes or worker/GPU replica
  ceilings in Phase 6, run Phase 8's burst tests against a staging
  environment first and size infra off real numbers rather than guesses.
