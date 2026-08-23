"use client";

/**
 * /admin/dashboard — take attendance for a section.
 *
 * Flow: pick section → upload/capture ONE classroom photo → the backend
 * enqueues a job and returns immediately → this page POLLS
 * GET /admin/session/capture/{job_id} (queued → processing → done) with a
 * progress indicator, then shows matched vs needs-review results.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError, tokenStore } from "@/lib/api";
import type { CaptureAccepted, CaptureStatus, Section } from "@/lib/types";
import CameraCapture from "@/components/CameraCapture";

const POLL_INTERVAL_MS = 1500;

type JobView = {
  accepted: CaptureAccepted;
  status: CaptureStatus | null;
  failed: boolean;
};

export default function AdminDashboardPage() {
  const router = useRouter();
  const [sections, setSections] = useState<Section[]>([]);
  const [sectionId, setSectionId] = useState("");
  const [courseName, setCourseName] = useState("");
  const [uploadedPhoto, setUploadedPhoto] = useState<Blob | null>(null);
  const [inputMode, setInputMode] = useState<"upload" | "camera">("upload");
  const [cameraShots, setCameraShots] = useState<Blob[]>([]);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<JobView | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!tokenStore.access || tokenStore.role === "student") {
      router.replace("/admin/login");
      return;
    }
    api.org
      .listSections()
      .then((rows) => {
        setSections(rows);
        if (rows.length === 1) setSectionId(rows[0].section_id);
      })
      .catch((e: ApiError) => setError(e.message));
  }, [router]);

  // Stop polling when leaving the page.
  useEffect(() => () => { if (pollTimer.current) clearTimeout(pollTimer.current); }, []);

  const poll = useCallback(
    (jobId: string) => {
      pollTimer.current = setTimeout(async () => {
        try {
          const status = await api.admin.getCaptureStatus(jobId);
          setJob((j) => (j ? { ...j, status } : j));
          if (status.status === "done" || status.status === "failed") return;
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) return; // status expired
          setJob((j) => (j ? { ...j, failed: true } : j));
          return;
        }
        poll(jobId);
      }, POLL_INTERVAL_MS);
    },
    []
  );

  // The photo actually submitted comes from whichever input mode is active.
  const photo = inputMode === "upload" ? uploadedPhoto : cameraShots[0] ?? null;

  const startCapture = async () => {
    if (!sectionId || !photo) return;
    setError("");
    setSubmitting(true);
    try {
      const accepted = await api.admin.capture(
        sectionId,
        courseName.trim() || "General",
        photo
      );
      setJob({ accepted, status: null, failed: false });
      poll(accepted.job_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Capture failed — please retry.");
    } finally {
      setSubmitting(false);
    }
  };

  const onPhotoPicked = (files: FileList | null) => {
    if (files?.length) setUploadedPhoto(files[0]);
  };

  const logout = () => {
    tokenStore.clear();
    router.push("/");
  };

  const stepIndex =
    job?.status?.status === "processing" ? 1 :
    job?.status?.status === "done" ? 2 :
    job?.failed ? -1 :
    job ? 0 : -1;

  return (
    <main className="page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 className="pageTitle">Take attendance</h1>
          <p className="pageSubtitle">Upload one classroom photo — results arrive in seconds.</p>
        </div>
        <nav className="btnRow">
          <a className="btn btnSecondary btnSm" href="/admin/review">Review queue</a>
          <a className="btn btnSecondary btnSm" href="/admin/reports">Reports</a>
          <a className="btn btnSecondary btnSm" href="/admin/bulk-enroll">Bulk enroll</a>
          {(tokenStore.role === "super_admin") && (
            <a className="btn btnSecondary btnSm" href="/admin/console">Console</a>
          )}
          <button className="btn btnSecondary btnSm" onClick={logout}>Log out</button>
        </nav>
      </div>

      {error && <div className="alert alertError">{error}</div>}

      <div className="grid2">
        <div className="card">
          <h2 className="cardTitle">1 · Session details</h2>
          <div className="field">
            <label className="label" htmlFor="section">Section</label>
            <select
              id="section"
              className="select"
              value={sectionId}
              onChange={(e) => setSectionId(e.target.value)}
            >
              <option value="">Select a section…</option>
              {sections.map((s) => (
                <option key={s.section_id} value={s.section_id}>
                  {s.name} ({s.academic_year})
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="label" htmlFor="course">Course name</label>
            <input
              id="course"
              className="input"
              value={courseName}
              onChange={(e) => setCourseName(e.target.value)}
              placeholder="DBMS"
            />
          </div>

          <h2 className="cardTitle">2 · Classroom photo</h2>
          <div className="btnRow" style={{ marginBottom: "0.75rem" }}>
            <button
              type="button"
              className={`btn ${inputMode === "upload" ? "btnPrimary" : "btnSecondary"}`}
              onClick={() => {
                setInputMode("upload");
                setCameraShots([]);
              }}
            >
              Upload photo
            </button>
            <button
              type="button"
              className={`btn ${inputMode === "camera" ? "btnPrimary" : "btnSecondary"}`}
              onClick={() => {
                setInputMode("camera");
                setUploadedPhoto(null);
              }}
            >
              Use camera
            </button>
          </div>

          {inputMode === "upload" ? (
            <>
              <label className="btn btnSecondary">
                Choose photo
                <input
                  type="file"
                  accept="image/jpeg,image/png"
                  hidden
                  onChange={(e) => onPhotoPicked(e.target.files)}
                />
              </label>
              {photo && (
                <p className="muted" style={{ marginTop: "0.5rem" }}>
                  Selected: {(photo as File).name ?? "photo"} ({(photo.size / 1024).toFixed(0)} KB)
                </p>
              )}
            </>
          ) : (
            <CameraCapture
              shots={cameraShots}
              onChange={setCameraShots}
              minShots={1}
              maxShots={1}
              subject="classroom photo"
            />
          )}

          <div style={{ marginTop: "1.25rem" }}>
            <button
              className="btn btnPrimary"
              disabled={!sectionId || !photo || submitting || !!job}
              onClick={startCapture}
            >
              {submitting ? "Uploading…" : "Start attendance"}
            </button>
          </div>
        </div>

        <div className="card">
          <h2 className="cardTitle">Processing status</h2>
          {!job && (
            <p className="muted">
              The capture runs asynchronously on a GPU worker pool — you can start
              the next class while this finishes.
            </p>
          )}
          {job && (
            <>
              <div className="progressSteps">
                {["Queued", "Processing", "Done"].map((label, idx) => (
                  <span
                    key={label}
                    className={`progressStep ${
                      idx === stepIndex ? "progressStepActive" : idx < stepIndex ? "progressStepDone" : ""
                    }`}
                  >
                    {label}
                  </span>
                ))}
              </div>
              <p className="mono muted">job_id: {job.accepted.job_id}</p>
              {job.failed && (
                <div className="alert alertError">
                  Lost contact with the job — check again from the review queue.
                </div>
              )}
              {job.status?.status === "failed" && (
                <div className="alert alertError">Processing failed: {job.status.error}</div>
              )}
              {job.status?.status === "done" && (
                <>
                  <div className="alert alertSuccess">
                    Done — {job.status.matched.length} matched,{" "}
                    {job.status.review_needed.length} need review.
                  </div>
                  {job.status.matched.length > 0 && (
                    <div className="tableWrap">
                      <table className="table">
                        <thead>
                          <tr><th>Matched students</th><th>Distance</th></tr>
                        </thead>
                        <tbody>
                          {job.status.matched.map((m) => (
                            <tr key={m.student_id}>
                              <td>{m.name ?? m.student_id}</td>
                              <td className="mono">{m.match_distance?.toFixed(3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {job.status.review_needed.length > 0 && (
                    <>
                      <div className="alert alertWarning">
                        {job.status.review_needed.length} face(s) were below the match
                        threshold and went to the review queue instead of being guessed.
                      </div>
                      <a className="btn btnWarning btn" href="/admin/review" style={{ background: "var(--warning)", color: "#fff" }}>
                        Open review queue
                      </a>
                    </>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </main>
  );
}
