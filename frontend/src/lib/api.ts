/**
 * src/lib/api.ts
 * ──────────────
 * THE single API client module. Every HTTP call to the FastAPI backend goes
 * through here — no page or component imports `fetch` directly. This keeps
 * auth headers, error parsing, and the base URL in one place.
 *
 * Auth model: JWT access + refresh tokens in localStorage, sent as Bearer
 * headers (CORS is configured for this). Phase 5 security hardening can move
 * these to httpOnly cookies behind Next.js route handlers without touching
 * any page — all call sites go through this module.
 */

import type {
  AttendanceReport,
  BulkEnrollResult,
  CaptureAccepted,
  CaptureStatus,
  Department,
  Institution,
  PublicSection,
  RegisterResponse,
  ReviewItem,
  ReviewResolved,
  RosterStudent,
  SearchAllCandidate,
  Section,
  StudentAttendanceResponse,
  TokenPair,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Token storage ─────────────────────────────────────────────────────────────

const ACCESS_KEY = "att.access_token";
const REFRESH_KEY = "att.refresh_token";
const ROLE_KEY = "att.role"; // "student" | admin role string

export const tokenStore = {
  get access(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(ACCESS_KEY);
  },
  get refresh(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(REFRESH_KEY);
  },
  get role(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(ROLE_KEY);
  },
  save(tokens: TokenPair, role: string): void {
    window.localStorage.setItem(ACCESS_KEY, tokens.access_token);
    window.localStorage.setItem(REFRESH_KEY, tokens.refresh_token);
    window.localStorage.setItem(ROLE_KEY, role);
  },
  clear(): void {
    window.localStorage.removeItem(ACCESS_KEY);
    window.localStorage.removeItem(REFRESH_KEY);
    window.localStorage.removeItem(ROLE_KEY);
  },
};

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ── Base fetch wrapper ────────────────────────────────────────────────────────

interface ApiRequestOptions {
  method?: string;
  body?: unknown; // JSON-serialised automatically
  formData?: FormData; // takes precedence over body when present
  skipAuth?: boolean;
}

async function apiFetch<T>(path: string, opts: ApiRequestOptions = {}): Promise<T> {
  const { method = "GET", body, formData, skipAuth = false } = opts;

  const headers: Record<string, string> = {};
  if (!formData && body !== undefined) headers["Content-Type"] = "application/json";
  const access = tokenStore.access;
  if (!skipAuth && access) headers["Authorization"] = `Bearer ${access}`;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: formData ?? (body !== undefined ? JSON.stringify(body) : undefined),
    });
  } catch {
    throw new ApiError(0, "Cannot reach the server — is the backend running?");
  }

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    // FastAPI validation errors arrive as {detail: [{msg...}, ...]}
    const detail = (err as { detail?: unknown }).detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => String((d as { msg?: string }).msg ?? d)).join("; ")
          : response.statusText || "Request failed";
    throw new ApiError(response.status, message);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function absoluteUrl(url: string | null): string | null {
  if (!url) return null;
  if (url.startsWith("http")) return url;
  return `${API_BASE_URL}${url}`;
}

// ─── Public endpoints ─────────────────────────────────────────────────────────

const pub = {
  health: () => apiFetch<{ status: string; environment: string }>("/health", { skipAuth: true }),
  listSections: () => apiFetch<PublicSection[]>("/public/sections", { skipAuth: true }),
};

// ─── Student endpoints ────────────────────────────────────────────────────────

const students = {
  login: async (rollNumber: string, password: string): Promise<TokenPair> => {
    const tokens = await apiFetch<TokenPair>("/auth/student/login", {
      method: "POST",
      body: { roll_number: rollNumber, password },
      skipAuth: true,
    });
    tokenStore.save(tokens, "student");
    return tokens;
  },

  register: async (
    fields: { rollNumber: string; name: string; password: string; sectionId: string; registrationToken?: string },
    images: Blob[]
  ): Promise<RegisterResponse> => {
    const fd = new FormData();
    fd.append("roll_number", fields.rollNumber);
    fd.append("name", fields.name);
    fd.append("password", fields.password);
    fd.append("section_id", fields.sectionId);
    if (fields.registrationToken) {
      fd.append("registration_token", fields.registrationToken.trim());
    }
    images.forEach((img, i) => fd.append("images", img, `enrollment_${i + 1}.jpg`));
    return apiFetch<RegisterResponse>("/register", { method: "POST", formData: fd, skipAuth: true });
  },

  myAttendance: () => apiFetch<StudentAttendanceResponse>("/attendance/me"),
};

// ─── Admin endpoints ──────────────────────────────────────────────────────────

const admin = {
  login: async (username: string, password: string): Promise<TokenPair> => {
    const tokens = await apiFetch<TokenPair>("/auth/admin/login", {
      method: "POST",
      body: { username, password },
      skipAuth: true,
    });
    // Role comes back inside the JWT claims; store it after decoding payload.
    try {
      const b64 = tokens.access_token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
      const payload = JSON.parse(atob(b64));
      tokenStore.save(tokens, String(payload.role ?? "admin"));
    } catch {
      tokenStore.save(tokens, "admin");
    }
    return tokens;
  },

  /** POST /admin/session/capture — enqueue-only; returns job_id immediately. */
  capture: async (sectionId: string, courseName: string, image: Blob): Promise<CaptureAccepted> => {
    const fd = new FormData();
    fd.append("section_id", sectionId);
    fd.append("course_name", courseName);
    fd.append("image", image, "capture.jpg");
    return apiFetch<CaptureAccepted>("/admin/session/capture", {
      method: "POST",
      formData: fd,
    });
  },

  getCaptureStatus: (jobId: string) =>
    apiFetch<CaptureStatus>(`/admin/session/capture/${jobId}`),

  listReviewQueue: (sectionId?: string, limit = 100) => {
    const q = new URLSearchParams({ limit: String(limit) });
    if (sectionId) q.set("section_id", sectionId);
    return apiFetch<ReviewItem[]>(`/admin/review-queue?${q.toString()}`);
  },

  resolveReview: (
    reviewId: string,
    action: "assign" | "discard",
    studentId?: string,
    searchAll = false
  ) =>
    apiFetch<ReviewResolved>(`/admin/review-queue/${reviewId}/resolve`, {
      method: "POST",
      body: { action, student_id: studentId ?? null, search_all: searchAll },
    }),

  searchAllForReview: (reviewId: string) =>
    apiFetch<{ review_id: string; candidates: SearchAllCandidate[] }>(
      `/admin/review-queue/${reviewId}/search-all`,
      { method: "POST" }
    ),

  getAttendanceReport: (sectionId: string, date: string) =>
    apiFetch<AttendanceReport>(
      `/admin/attendance?section_id=${encodeURIComponent(sectionId)}&date=${encodeURIComponent(date)}`
    ),

  bulkEnrollCsv: async (sectionId: string, csvFile: File): Promise<BulkEnrollResult> => {
    const fd = new FormData();
    fd.append("file", csvFile, csvFile.name);
    return apiFetch<BulkEnrollResult>(`/admin/sections/${sectionId}/bulk-enroll`, {
      method: "POST",
      formData: fd,
    });
  },

  listStudentsInSection: (sectionId: string) =>
    apiFetch<RosterStudent[]>(`/admin/sections/${sectionId}/students`),

  issueRegistrationToken: (rollNumber: string, expiresInDays = 30) =>
    apiFetch<{ token: string; roll_number: string; expires_at: string }>(
      "/admin/registration-tokens",
      {
        method: "POST",
        body: { roll_number: rollNumber, expires_in_days: expiresInDays },
      }
    ),
};

// ─── Org CRUD (super-admin / scoped admins) ──────────────────────────────────

const org = {
  listInstitutions: () => apiFetch<Institution[]>("/admin/institutions"),
  createInstitution: (name: string) =>
    apiFetch<Institution>("/admin/institutions", { method: "POST", body: { name } }),

  listDepartments: (institutionId?: string) => {
    const suffix = institutionId ? `?institution_id=${institutionId}` : "";
    return apiFetch<Department[]>(`/admin/departments${suffix}`);
  },
  createDepartment: (institutionId: string, name: string) =>
    apiFetch<Department>("/admin/departments", {
      method: "POST",
      body: { institution_id: institutionId, name },
    }),

  listSections: (departmentId?: string, institutionId?: string) => {
    const q = new URLSearchParams();
    if (departmentId) q.set("department_id", departmentId);
    if (institutionId) q.set("institution_id", institutionId);
    const qs = q.toString();
    return apiFetch<Section[]>(`/admin/sections${qs ? `?${qs}` : ""}`);
  },
  createSection: (departmentId: string, name: string, academicYear: number) =>
    apiFetch<Section>("/admin/sections", {
      method: "POST",
      body: { department_id: departmentId, name, academic_year: academicYear },
    }),

  createAdminAccount: (body: {
    username: string;
    password: string;
    role: string;
    institution_id: string | null;
    department_id: string | null;
  }) => apiFetch<{ admin_id: string; username: string; role: string }>("/admin/admins", {
    method: "POST",
    body,
  }),
};

// ─── Default export ───────────────────────────────────────────────────────────

const api = { pub, students, admin, org, absoluteUrl, ApiError, tokenStore };
export default api;
