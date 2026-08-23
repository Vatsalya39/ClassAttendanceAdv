# docs/architecture.md
# System Architecture

## Overview

Multi-campus face-recognition attendance system supporting up to 50,000 enrolled
students across many institutions, departments, and sections.

## The Single Most Important Design Rule

> **Face matching for a class session is ALWAYS scoped to that section's enrolled
> roster (~60-70 candidates), never a global search across all enrolled students.**

This is what keeps match latency flat whether you have 5,000 or 50,000 students.
The pgvector HNSW index is never queried unfiltered. Every call to the vector index
is preceded by a join to `enrollments` filtered by `section_id`, reducing the
candidate set to ~60-70 embeddings regardless of total enrollment.

The one legitimate broader search — a faculty admin manually identifying an
unmatched face from the review queue — uses `find_nearest_embeddings_institution_wide`
which is explicitly named, expected to be slower, and used rarely.

## Organisational Hierarchy

```
Institution
└── Department
    └── Section (~60-70 students)
        └── Class Session
            └── Attendance / Review Queue
```

## Async Capture Flow

```
Admin uploads photo
        │
        ▼
POST /admin/session/capture
  ├─ Validate admin scope (must own the section)
  ├─ Store raw image to Blob Storage
  ├─ Enqueue CaptureJobMessage {job_id, session_id, section_id, image_url}
  └─ Return {job_id, status: "queued"} immediately
        │
        ▼ (async, separate process)
Worker consumes job
  ├─ batch_process(image) → list of embeddings
  ├─ For each embedding:
  │   └─ find_nearest_embeddings_for_roster(embedding, section_id)
  │       ├─ Match within threshold → INSERT attendance
  │       └─ No match             → INSERT review_queue
  └─ Update job status in Redis
        │
        ▼
Admin polls GET /admin/session/capture/{job_id}
  └─ {status: done, matched: [...], review_needed: [...]}
```

## Admin Role Hierarchy

| Role | Scope |
|---|---|
| `super_admin` | All institutions |
| `institution_admin` | One institution |
| `department_admin` | One department within an institution |
| `faculty` | Sections assigned to them |

JWT claims carry `institution_id` and `department_id`. Every admin route
validates the claim against the target resource — a department_admin for
Dept A gets 403 touching Dept B's sections.

## Key Diagrams

### Async capture flow (sequence)

```mermaid
sequenceDiagram
    autonumber
    actor Admin
    participant API as FastAPI (api tier)
    participant Q as Queue (Redis / Service Bus)
    participant W as Worker (KEDA-scaled)
    participant ML as ml.pipeline (warm model)
    participant DB as PostgreSQL + pgvector
    participant R as Redis

    Admin->>API: POST /admin/session/capture (photo, section_id)
    API->>API: verify admin scope + rate limit
    API->>DB: INSERT class_session
    API->>R: SET job:{id} = queued
    API->>Q: ENQUEUE CaptureJobMessage
    API-->>Admin: 200 {job_id, status: "queued"}  ← returns immediately

    W->>Q: consume message
    W->>R: SET NX job:{id}:done (idempotency claim)
    W->>ML: batch_process(image)
    ML-->>W: [{bbox, 512-d embedding}, …]
    loop for each detected face
        W->>DB: find_nearest_embeddings_for_roster(embedding, section_id)
        Note over DB: joins enrollments FIRST —<br/>~60-70 candidates, never global
        alt distance ≤ threshold
            W->>DB: INSERT attendance (ON CONFLICT DO NOTHING)
        else above threshold / no roster
            W->>DB: INSERT review_queue (crop + embedding + best guess)
        end
    end
    W->>R: SET job:{id} = done {matched, review_needed}
    W->>Q: ack

    Admin->>API: GET /admin/session/capture/{job_id}
    API->>R: GET job:{id}
    API-->>Admin: {status: done, matched[…], review_needed[…]}
```

### Roster-scoped matching rule

```mermaid
flowchart LR
    A[face embedding] --> B{search type?}
    B -->|"automatic capture<br/>(the ONLY allowed path)"| C["find_nearest_embeddings_for_roster<br/>JOIN enrollments ON section_id"]
    C --> D["~60-70 students × 3-5 images<br/>= few hundred candidates<br/>latency flat at ANY total enrollment"]
    B -->|"manual review 'search all'<br/>(explicit, rare)"| E["find_nearest_embeddings_institution_wide<br/>scoped to ONE institution"]
    D --> F[pgvector HNSW index<br/>never queried unfiltered]
    E --> F
```

### Entity relationships

```mermaid
erDiagram
    institutions ||--o{ departments : has
    departments ||--o{ sections : has
    institutions ||--o{ students : has
    students ||--o{ enrollments : ""
    sections ||--o{ enrollments : ""
    sections ||--o{ class_sessions : ""
    students ||--o{ face_embeddings : "3-5 each"
    class_sessions ||--o{ attendance : ""
    students ||--o{ attendance : ""
    admins ||--o{ class_sessions : captures
    class_sessions ||--o{ review_queue : ""
    admins ||--o{ audit_log : ""

    attendance {
        date session_date PK "partition key (monthly RANGE)"
        uuid attendance_id PK
        uuid student_id
        uuid session_id
        float match_distance
    }
```

Also see the `ClassAttendenceResearch/Architecture/` directory for:
- System sequence diagram (async capture flow)
- Architecture block diagram
- Database ER diagram
- Internal service diagram

## Database Partitioning

The `attendance` table is partitioned by range on `session_date` (monthly).
At 50,000 students × 200 session-days/year ≈ 10M rows/year. Partitioning
ensures reporting queries only scan relevant months and vacuum overhead is
bounded per-partition.

A scheduled maintenance job pre-creates the next 3 months of partitions
to avoid missing-partition errors on first insert.

## Scalability Model

| Tier | Scale trigger |
|---|---|
| API | HTTP concurrency (Container Apps built-in) |
| Worker | Azure Service Bus message count (KEDA) |
| ML Inference | Queue depth (KEDA), min=0, scales to zero when idle |
| Database | Read replica for analytics, primary for writes |
| Cache | Redis for roster cache, rate limiting, job status |
