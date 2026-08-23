/**
 * src/lib/types.ts
 * ────────────────
 * Shared TypeScript types mirroring the backend's Pydantic response models
 * (backend/api/schemas.py). Keep in sync with the API.
 */

export type AdminRole =
  | "super_admin"
  | "institution_admin"
  | "department_admin"
  | "faculty";

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface RegisterResponse {
  student_id: string;
  roll_number: string;
  name: string;
  embeddings_stored: number;
}

export interface AttendanceRecord {
  session_id: string;
  course_name: string;
  session_date: string;
  marked_at: string | null;
  match_distance: number | null;
}

export interface StudentAttendanceResponse {
  student_id: string;
  records: AttendanceRecord[];
}

export interface PublicSection {
  section_id: string;
  name: string;
  academic_year: number;
  department_name: string;
  institution_name: string;
}

export interface Section {
  section_id: string;
  department_id: string;
  name: string;
  academic_year: number;
}

export interface RosterStudent {
  student_id: string;
  roll_number: string;
  name: string;
}

export interface CaptureAccepted {
  job_id: string;
  session_id: string;
  status: "queued" | "processing" | "done" | "failed";
}

export interface CaptureResultStudent {
  student_id: string;
  roll_number: string | null;
  name: string | null;
  match_distance: number | null;
}

export interface CaptureStatus {
  job_id: string;
  status: "queued" | "processing" | "done" | "failed";
  error?: string | null;
  matched: CaptureResultStudent[];
  review_needed: CaptureResultStudent[];
}

export interface ReviewItem {
  review_id: string;
  session_id: string;
  section_id: string;
  section_name: string;
  face_crop_url: string | null;
  best_guess_student_id: string | null;
  best_guess_distance: number | null;
  created_at: string;
}

export interface ReviewResolved {
  review_id: string;
  status: string;
  resolved_student_id: string | null;
}

export interface SearchAllCandidate {
  student_id: string;
  roll_number: string;
  name: string;
  distance: number;
}

export interface AttendanceReportRow {
  student_id: string;
  roll_number: string;
  name: string;
  marked: boolean;
  match_distance: number | null;
  marked_at: string | null;
}

export interface AttendanceReport {
  section_id: string;
  date: string;
  total_enrolled: number;
  present: number;
  rows: AttendanceReportRow[];
}

export interface Institution {
  institution_id: string;
  name: string;
}

export interface Department {
  department_id: string;
  institution_id: string;
  name: string;
}

export interface BulkEnrollResult {
  section_id: string;
  enrolled: number;
  skipped: number;
}
