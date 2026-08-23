# docs/data-retention.md
# Biometric Data Retention Policy

## Overview

This system processes and stores biometric data — specifically face images and
512-dimensional face embeddings (derived biometric templates). This document
describes data handling practices, retention limits, and deletion procedures
required under applicable biometric privacy laws (e.g. BIPA, GDPR Art. 9,
India DPDP Act, and equivalent state/national regulations).

> **Note**: This is a technical policy stub. Before production deployment,
> this document must be reviewed and approved by legal counsel for the relevant
> jurisdictions. It is NOT a substitute for a formal privacy notice or DPA.

## What Biometric Data We Store

| Data type | Storage location | Retention |
|---|---|---|
| Enrollment photos (raw JPEGs) | Azure Blob Storage (`enrollment-images` container) | Until student deletion request |
| Session capture photos (raw JPEGs) | Azure Blob Storage (`session-images` container) | 90 days after session date |
| Face embeddings (512-d float vectors) | PostgreSQL `face_embeddings` table | Until student deletion request |
| Attendance records (session presence + match distance) | PostgreSQL `attendance` partitioned table | Per institution policy (default: academic year + 2 years) |
| Review queue face crops | Azure Blob Storage + `review_queue` table | 30 days after resolution or 90 days maximum |

## Consent

Face registration (enrollment) requires:
1. Explicit, informed written consent from the student prior to capture.
2. Disclosure of: purpose, retention period, deletion rights, and who can access the data.
3. A consent record stored independently of the biometric data (not in this system — handled by the institution's LMS or HR system).

Students under the age of consent require guardian approval (institution responsibility).

## Right to Erasure (Student Deletion Request)

**Implemented (Phase 5):** `POST /admin/students/{id}/purge-biometrics`
(super_admin / institution_admin of the student's institution only). Body:
`{"attendance_action": "anonymize" | "delete", "delete_account": bool}`.
It performs all five steps below atomically and writes an audit entry;
attendance anonymisation is enabled by migration
`0002_attendance_anonymizable` (`attendance.student_id` is NULLable).

When a student requests deletion of their biometric data:

1. **Delete all face embeddings** for the student:
   ```sql
   DELETE FROM face_embeddings WHERE student_id = :student_id;
   ```
2. **Delete enrollment images** from Blob Storage:
   - Enumerate `source_image_url` values before deletion.
   - Call `StorageBackend.delete(url)` for each.
3. **Anonymise attendance records** (retain for statistical/audit purposes but
   strip the student reference):
   ```sql
   UPDATE attendance
   SET student_id = NULL,   -- or a placeholder UUID
       match_distance = NULL
   WHERE student_id = :student_id;
   ```
   OR delete entirely — institution policy decision.
4. **Delete review queue items** referencing the student:
   ```sql
   DELETE FROM review_queue WHERE best_guess_student_id = :student_id;
   ```
5. Write an audit log entry for the deletion action.

**Phase 5 note:** all five steps are implemented in
`backend/api/routes/admin.py::purge_biometrics`. Blob-storage failures are
logged but never block database erasure (a storage lifecycle policy cleans
orphans in production).

## Session Capture Photos

Raw classroom photos are stored temporarily for audit and review-queue purposes.
Automatic deletion of session capture blobs after 90 days is handled by an Azure
Blob Storage lifecycle management policy (configured in Phase 6 infra).

## Access Control

| Data | Who can read |
|---|---|
| Raw enrollment photos | institution_admin, super_admin only |
| Face embeddings | ML worker (write at enrollment, read at match time) — no human-readable access |
| Attendance records | faculty for their sections, institution_admin, super_admin |
| Review queue crops | faculty/department_admin for their sections only |

No face image or embedding is ever returned in an API response to a student or
unauthenticated caller.

## Security at Rest and in Transit

- All data in transit: TLS 1.2+ enforced by Azure Front Door / Container Apps ingress.
- Blob Storage: encryption at rest enabled by default (Microsoft-managed keys;
  consider Customer-Managed Keys for regulated institutions).
- PostgreSQL: transparent data encryption enabled on Azure Database for PostgreSQL
  Flexible Server.
- Embeddings are not encrypted at the column level (they are numeric arrays, not
  personally identifiable in isolation), but the database itself is encrypted at
  rest and not publicly accessible.

## Audit Trail

Every admin action touching biometric data is written to the `audit_log` table:
- Enrollment (image upload + embedding creation)
- Deletion / purge requests
- Review queue resolution
- Bulk enrollment imports

Audit logs are retained for 7 years (configurable per institution) and are
not deletable via the API — only via a separate DBA procedure with super_admin
approval recorded externally.
