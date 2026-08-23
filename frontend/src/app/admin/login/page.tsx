"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError } from "@/lib/api";

export default function AdminLoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await api.admin.login(username.trim(), password);
      router.push("/admin/dashboard");
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "Invalid username or password."
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
      <h1 className="pageTitle">Admin login</h1>
      <p className="pageSubtitle">Faculty, department and institution admins</p>
      <form className="card" onSubmit={submit}>
        <div className="field">
          <label className="label" htmlFor="username">
            Username
          </label>
          <input
            id="username"
            className="input"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
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
