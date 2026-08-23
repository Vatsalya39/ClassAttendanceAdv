"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError } from "@/lib/api";

export default function StudentLoginPage() {
  const router = useRouter();
  const [rollNumber, setRollNumber] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await api.students.login(rollNumber.trim(), password);
      router.push("/student/attendance");
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "Invalid roll number or password."
          : err instanceof ApiError
            ? err.message
            : "Login failed. Please try again."
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="page pageNarrow">
      <h1 className="pageTitle">Student login</h1>
      <p className="pageSubtitle">Check your attendance history</p>
      <form className="card" onSubmit={submit}>
        <div className="field">
          <label className="label" htmlFor="roll">
            Roll number
          </label>
          <input
            id="roll"
            className="input"
            value={rollNumber}
            onChange={(e) => setRollNumber(e.target.value)}
            autoFocus
          />
        </div>
        <div className="field">
          <label className="label" htmlFor="password">
            Password
          </label>
          <input
            id="password"
            type="password"
            className="input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        {error && <div className="alert alertError">{error}</div>}
        <button className="btn btnPrimary" type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Log in"}
        </button>
      </form>
    </main>
  );
}
