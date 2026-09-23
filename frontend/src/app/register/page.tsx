"use client";

/**
 * /register — student enrollment flow.
 * Step 1: details (name, roll number, password, section)
 * Step 2: guided camera capture of 3-5 shots → POST /register
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import api, { ApiError } from "@/lib/api";
import type { PublicSection } from "@/lib/types";
import CameraCapture, { type PoseStep } from "@/components/CameraCapture";

const ENROLLMENT_POSES: PoseStep[] = [
  {
    id: "frontal",
    label: "Frontal",
    instruction: "Look straight at the camera with a neutral expression and eyes open.",
    badge: "Frontal (0°)",
  },
  {
    id: "left",
    label: "Slight Left",
    instruction: "Turn head gently ~15° to your left, keeping eyes towards the camera.",
    badge: "Left (~15°)",
  },
  {
    id: "right",
    label: "Slight Right",
    instruction: "Turn head gently ~15° to your right, keeping eyes towards the camera.",
    badge: "Right (~15°)",
  },
];

export default function RegisterPage() {
  const [step, setStep] = useState<1 | 2>(1);
  const [sections, setSections] = useState<PublicSection[]>([]);
  const [loadError, setLoadError] = useState("");

  const [rollNumber, setRollNumber] = useState("");
  const [name, setName] = useState("");
  const [registrationToken, setRegistrationToken] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [sectionId, setSectionId] = useState("");

  const [shots, setShots] = useState<Blob[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ name: string; stored: number } | null>(null);

  useEffect(() => {
    api.pub
      .listSections()
      .then(setSections)
      .catch((e: ApiError) => setLoadError(e.message));
  }, []);

  const detailsValid =
    rollNumber.trim().length > 0 &&
    name.trim().length > 0 &&
    registrationToken.trim().length > 0 &&
    password.length >= 8 &&
    password === confirmPassword &&
    sectionId !== "";

  const submit = async () => {
    setError("");
    setSubmitting(true);
    try {
      const res = await api.students.register(
        {
          rollNumber: rollNumber.trim(),
          name: name.trim(),
          password,
          sectionId,
          registrationToken: registrationToken.trim(),
        },
        shots
      );
      setResult({ name: res.name, stored: res.embeddings_stored });
      api.tokenStore.clear(); // registration does not log the student in
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.status === 409
            ? "This roll number is already registered — try logging in instead."
            : e.status === 403
            ? "Registration token is invalid, expired, or already used. Please contact your instructor."
            : e.message
          : "Registration failed. Please try again."
      );
      // Return to the capture step so the user can retry without retyping.
      if (shots.length > 0) setStep(2);
    } finally {
      setSubmitting(false);
    }
  };

  if (result) {
    return (
      <main className="page pageNarrow">
        <h1 className="pageTitle">You&apos;re enrolled 🎉</h1>
        <div className="card">
          <div className="alert alertSuccess">
            Welcome, {result.name}! {result.stored} face embedding
            {result.stored === 1 ? "" : "s"} were stored securely.
          </div>
          <p className="muted">
            From now on, just attend class — your teacher&apos;s classroom photo marks
            your attendance automatically.
          </p>
          <div className="btnRow" style={{ marginTop: "1rem" }}>
            <Link className="btn btnPrimary" href="/student/login">
              Log in to view attendance
            </Link>
            <Link className="btn btnSecondary" href="/">
              Back home
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="page pageNarrow">
      <h1 className="pageTitle">Register with your face</h1>
      <p className="pageSubtitle">
        Step {step} of 2 — {step === 1 ? "your details" : "guided camera captures"}
      </p>

      {loadError && <div className="alert alertError">{loadError}</div>}
      {!loadError && sections.length === 0 && step === 1 && (
        <div className="alert alertInfo">Loading sections…</div>
      )}

      {step === 1 && (
        <div className="card">
          <div className="field">
            <label className="label" htmlFor="name">
              Full name
            </label>
            <input
              id="name"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Ada Lovelace"
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="roll">
              Roll number
            </label>
            <input
              id="roll"
              className="input"
              value={rollNumber}
              onChange={(e) => setRollNumber(e.target.value)}
              placeholder="CSE3B042"
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="section">
              Section
            </label>
            <select
              id="section"
              className="select"
              value={sectionId}
              onChange={(e) => setSectionId(e.target.value)}
            >
              <option value="">Select your section…</option>
              {sections.map((s) => (
                <option key={s.section_id} value={s.section_id}>
                  {s.institution_name} · {s.department_name} · {s.name} ({s.academic_year})
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="label" htmlFor="regToken">
              Registration token / invitation code
            </label>
            <input
              id="regToken"
              className="input mono"
              value={registrationToken}
              onChange={(e) => setRegistrationToken(e.target.value)}
              placeholder="e.g. REG-A1B2-C3D4 or token from instructor"
              autoComplete="off"
            />
            <span className="muted" style={{ fontSize: "0.8rem", marginTop: "0.25rem" }}>
              Required single-use token provided by your department or faculty.
            </span>
          </div>
          <div className="field">
            <label className="label" htmlFor="password">
              Password (min 8 characters)
            </label>
            <input
              id="password"
              type="password"
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="password2">
              Confirm password
            </label>
            <input
              id="password2"
              type="password"
              className="input"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
          </div>
          {password && confirmPassword && password !== confirmPassword && (
            <div className="alert alertError">Passwords do not match.</div>
          )}
          <button
            className="btn btnPrimary"
            disabled={!detailsValid}
            onClick={() => setStep(2)}
          >
            Continue to face capture
          </button>
        </div>
      )}

      {step === 2 && (
        <div className="card">
          <CameraCapture
            shots={shots}
            onChange={setShots}
            minShots={3}
            maxShots={5}
            subject="face angles"
            poseSteps={ENROLLMENT_POSES}
          />
          {error && <div className="alert alertError">{error}</div>}
          <div className="btnRow" style={{ marginTop: "1rem" }}>
            <button className="btn btnSecondary" onClick={() => setStep(1)}>
              Back
            </button>
            <button
              className="btn btnPrimary"
              disabled={shots.length < 3 || submitting}
              onClick={submit}
            >
              {submitting ? "Processing…" : `Register with ${shots.length} photo${shots.length === 1 ? "" : "s"}`}
            </button>
          </div>
          <p
            className="muted"
            style={{
              marginTop: "1.25rem",
              borderTop: "1px solid var(--border)",
              paddingTop: "0.85rem",
              lineHeight: 1.5,
            }}
          >
            💡 <strong>Why multiple angles?</strong> Capturing your face from frontal and slight angles creates
            robust reference embeddings, allowing automated attendance to recognize you accurately in class
            from any seat without runtime template drift.
          </p>
        </div>
      )}
    </main>
  );
}
