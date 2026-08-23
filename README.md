# Multi-Campus Face Recognition Attendance System

A production-grade attendance system supporting up to 50,000 enrolled students
across institutions, departments, and sections. Face matching is always scoped
to a section's enrolled roster (~60-70 students) — never a global search.

## Architecture overview

See [docs/architecture.md](docs/architecture.md) for the full design,
including the roster-scoping rule, async capture flow, and role hierarchy.

## Local development setup

### Prerequisites

- Docker Desktop (or Colima on macOS)
- Node.js 20+ (for frontend)
- Python 3.11+ (for backend / scripts)
- Git

### 1. Clone and configure

```bash
git clone <repo-url>
cd classAttendenceAdvanced
cp .env.example .env
```

Edit `.env` and set at minimum:
- `JWT_SECRET_KEY` — generate with `python -c "import secrets; print(secrets.token_hex(64))"`

All other defaults in `.env.example` work for local docker-compose.

### 2. Start all services

```bash
docker compose up --build
```

This starts:
| Service | URL |
|---|---|
| PostgreSQL (pgvector) | `localhost:5432` |
| Redis | `localhost:6379` |
| FastAPI API | http://localhost:8000 |
| Capture Worker | (background, no HTTP) |
| Next.js Frontend | http://localhost:3000 |

### 3. Run database migrations (Phase 1+)

```bash
docker compose exec api alembic upgrade head
```

### 4. Verify everything is up

```bash
curl http://localhost:8000/health
# {"status":"ok","environment":"development"}

open http://localhost:3000
```

### 5. Backend development (without Docker)

```bash
pip install poetry
poetry install
pre-commit install
uvicorn backend.api.main:app --reload
alembic upgrade head   # apply migrations (requires postgres from docker-compose)
```

### 6. Frontend development (without Docker)

```bash
cd frontend
npm install
npm run dev
```

### 7. Tests

```bash
# Full suite — DB tests auto-skip unless Postgres/Redis are reachable:
docker compose up -d postgres redis
poetry run pytest

# Lint + type check
poetry run ruff check backend ml scripts tests
poetry run mypy backend
```

DB-backed tests use a dedicated `attendance_test` database that is created
automatically from the compose Postgres — your app database is never touched.
Override with `TEST_DATABASE_URL` / `TEST_REDIS_URL` for CI.

## Environment variables

All variables are documented in [.env.example](.env.example).

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | Async SQLAlchemy DSN (asyncpg) |
| `DATABASE_URL_SYNC` | ✅ | Sync DSN for Alembic |
| `JWT_SECRET_KEY` | ✅ | ≥32 char random secret |
| `REDIS_URL` | ✅ | Redis connection string |
| `QUEUE_BACKEND` | ✅ | `redis` (local) or `servicebus` (prod) |
| `STORAGE_BACKEND` | ✅ | `local` (local) or `azure_blob` (prod) |
| `FACE_MATCH_THRESHOLD` | ✅ | Cosine distance threshold (calibrate in Phase 8) |
| `SERVICE_BUS_CONNECTION_STRING` | prod only | Azure Service Bus |
| `AZURE_STORAGE_CONNECTION_STRING` | prod only | Azure Blob Storage |
| `AZURE_KEYVAULT_URI` | prod only | Azure Key Vault URI |

## Project structure

```
.
├── backend/
│   ├── api/          # FastAPI HTTP layer (routes, schemas, deps)
│   ├── worker/       # Queue consumer (separate process)
│   ├── core/         # Config, queue, storage, security, rate limit, job status
│   ├── services/     # Business logic shared by API + worker (capture flow)
│   ├── db/           # SQLAlchemy models + DAO + partition helpers (Phase 1)
│   └── alembic/      # Database migrations
├── frontend/         # Next.js 14 (App Router) — Phase 4
├── ml/               # Face pipeline module (Phase 2)
├── scripts/          # CLI tools (calibrate, bench, test_pipeline)
├── infra/            # Bicep IaC (Phase 6)
├── docs/             # Architecture, data retention policy
├── docker-compose.yml
├── .env.example
└── alembic.ini
```

## Build phases

| Phase | Description |
|---|---|
| **0** ✅ | Repo & environment scaffold |
| **1** ✅ | Database layer (SQLAlchemy models + Alembic + roster-scoped DAO + partitioned attendance) |
| **2** ✅ | Face pipeline module (InsightFace RetinaFace + ArcFace) |
| **3** ✅ | FastAPI endpoints + async capture worker (JWT auth, rate limiting, review queue, tenant isolation) |
| **4** ✅ | React/Next.js frontend (register w/ camera capture, student & admin portals, review queue, reports, super-admin console) |
| **5** ✅ | Security hardening (argon2id params, register rate limit, security headers, scoped job polling, tenant-isolation test suite, purge-biometrics endpoint) |
| 6 | Azure IaC (Bicep) |
| 7 | CI/CD (GitHub Actions) |
| 8 | Accuracy calibration + load testing |

## Key design decisions

1. **Roster-scoped matching** — `find_nearest_embeddings_for_roster(vector, section_id)`
   joins to `enrollments` before the ANN search. ~60-70 candidates, not 50,000.
2. **Async capture** — the capture endpoint enqueues a job and returns `job_id`
   immediately. Workers scale independently via KEDA.
3. **Pluggable queue/storage** — `QueueBackend` and `StorageBackend` protocols
   let local dev use Redis + local FS; production uses Azure Service Bus + Blob.
4. **Monthly attendance partitions** — the `attendance` table is partitioned by
   `session_date` for query performance at 10M+ rows/year.

## Scripts

```bash
# Create the first super admin (needed before any /admin CRUD works)
poetry run python scripts/create_superadmin.py --username admin --password <min-8-chars>

# Smoke-test the face pipeline on one image
python scripts/test_pipeline.py path/to/image.jpg

# Calibrate the match threshold on labeled pairs
python scripts/calibrate_threshold.py --pairs-dir data/calibration_pairs

# Benchmark GPU throughput and estimate KEDA replica count
python scripts/bench_throughput.py --image-dir data/bench_images
```

## Contributing

- Backend linting: `pre-commit run --all-files` (black + ruff + isort + mypy)
- Tests: `cd backend && pytest`
- Do not bypass the roster-scoping rule — any `find_nearest_embeddings` call
  without a `section_id` filter is a bug.
