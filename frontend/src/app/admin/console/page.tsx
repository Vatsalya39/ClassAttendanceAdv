"use client";

/**
 * /admin/console — super-admin console.
 * Create/manage institutions, departments, sections, and admin accounts.
 * The backend enforces role scope; this page simply hides the console link
 * for non-super-admins (and the backend rejects them anyway).
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError, tokenStore } from "@/lib/api";
import type { Department, Institution, Section } from "@/lib/types";

export default function SuperAdminConsolePage() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const [institutions, setInstitutions] = useState<Institution[]>([]);
  const [departments, setDepartments] = useState<Department[]>([]);
  const [sections, setSections] = useState<Section[]>([]);

  const [instName, setInstName] = useState("");
  const [deptName, setDeptName] = useState("");
  const [deptInstId, setDeptInstId] = useState("");
  const [sectionName, setSectionName] = useState("");
  const [sectionDeptId, setSectionDeptId] = useState("");
  const [sectionYear, setSectionYear] = useState(new Date().getFullYear());

  const [adminUsername, setAdminUsername] = useState("");
  const [adminPassword, setAdminPassword] = useState("");
  const [adminRole, setAdminRole] = useState("department_admin");
  const [adminInstId, setAdminInstId] = useState("");
  const [adminDeptId, setAdminDeptId] = useState("");

  const [tokenRollNumber, setTokenRollNumber] = useState("");
  const [tokenDays, setTokenDays] = useState(30);
  const [issuedToken, setIssuedToken] = useState<{ token: string; roll: string; expires: string } | null>(null);
  const [issuingToken, setIssuingToken] = useState(false);
  const [tokenCopied, setTokenCopied] = useState(false);

  const refreshAll = useCallback(async () => {
    try {
      const [insts, depts, secs] = await Promise.all([
        api.org.listInstitutions(),
        api.org.listDepartments(),
        api.org.listSections(),
      ]);
      setInstitutions(insts);
      setDepartments(depts);
      setSections(secs);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        tokenStore.clear();
        router.replace("/admin/login");
      } else if (e instanceof ApiError && e.status === 403) {
        setError("This console is only available to super admins.");
      } else {
        setError(e instanceof ApiError ? e.message : "Failed to load data");
      }
    }
  }, [router]);

  useEffect(() => {
    if (!tokenStore.access || tokenStore.role === "student") {
      router.replace("/admin/login");
      return;
    }
    void refreshAll();
  }, [router, refreshAll]);

  const run = async (fn: () => Promise<unknown>, okMessage: string) => {
    setError("");
    setNotice("");
    try {
      await fn();
      setNotice(okMessage);
      await refreshAll();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Request failed");
    }
  };

  return (
    <main className="page">
      <h1 className="pageTitle">Super-admin console</h1>
      <p className="pageSubtitle">
        Manage institutions, departments, sections and admin accounts
      </p>

      {error && <div className="alert alertError">{error}</div>}
      {notice && <div className="alert alertSuccess">{notice}</div>}

      <div className="grid2">
        <div className="card">
          <h2 className="cardTitle">Institutions</h2>
          <ul className="muted" style={{ listStyle: "none", marginBottom: "0.75rem" }}>
            {institutions.map((i) => (
              <li key={i.institution_id}>• {i.name}</li>
            ))}
          </ul>
          <div className="btnRow">
            <input
              className="input"
              style={{ flex: 1, minWidth: 160 }}
              placeholder="New institution name"
              value={instName}
              onChange={(e) => setInstName(e.target.value)}
            />
            <button
              className="btn btnPrimary"
              disabled={!instName.trim()}
              onClick={() =>
                run(() => api.org.createInstitution(instName.trim()), "Institution created")
              }
            >
              Create
            </button>
          </div>
        </div>

        <div className="card">
          <h2 className="cardTitle">Departments</h2>
          <ul className="muted" style={{ listStyle: "none", marginBottom: "0.75rem" }}>
            {departments.map((d) => (
              <li key={d.department_id}>
                • {d.name} (
                {institutions.find((i) => i.institution_id === d.institution_id)?.name ??
                  d.institution_id.slice(0, 8)}
                )
              </li>
            ))}
          </ul>
          <div className="btnRow" style={{ marginBottom: "0.5rem" }}>
            <select
              className="select"
              style={{ flex: 1 }}
              value={deptInstId}
              onChange={(e) => setDeptInstId(e.target.value)}
            >
              <option value="">Institution…</option>
              {institutions.map((i) => (
                <option key={i.institution_id} value={i.institution_id}>
                  {i.name}
                </option>
              ))}
            </select>
          </div>
          <div className="btnRow">
            <input
              className="input"
              style={{ flex: 1, minWidth: 160 }}
              placeholder="New department name"
              value={deptName}
              onChange={(e) => setDeptName(e.target.value)}
            />
            <button
              className="btn btnPrimary"
              disabled={!deptName.trim() || !deptInstId}
              onClick={() =>
                run(
                  () => api.org.createDepartment(deptInstId, deptName.trim()),
                  "Department created"
                )
              }
            >
              Create
            </button>
          </div>
        </div>

        <div className="card">
          <h2 className="cardTitle">Sections</h2>
          <ul className="muted" style={{ listStyle: "none", marginBottom: "0.75rem" }}>
            {sections.map((s) => (
              <li key={s.section_id}>• {s.name} ({s.academic_year})</li>
            ))}
          </ul>
          <div className="btnRow" style={{ marginBottom: "0.5rem" }}>
            <select
              className="select"
              style={{ flex: 1 }}
              value={sectionDeptId}
              onChange={(e) => setSectionDeptId(e.target.value)}
            >
              <option value="">Department…</option>
              {departments.map((d) => (
                <option key={d.department_id} value={d.department_id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>
          <div className="btnRow">
            <input
              className="input"
              placeholder="CSE-3B"
              value={sectionName}
              onChange={(e) => setSectionName(e.target.value)}
              style={{ width: 120 }}
            />
            <input
              className="input"
              type="number"
              value={sectionYear}
              onChange={(e) => setSectionYear(Number(e.target.value))}
              style={{ width: 100 }}
            />
            <button
              className="btn btnPrimary"
              disabled={!sectionName.trim() || !sectionDeptId}
              onClick={() =>
                run(
                  () => api.org.createSection(sectionDeptId, sectionName.trim(), sectionYear),
                  "Section created"
                )
              }
            >
              Create
            </button>
          </div>
        </div>

        <div className="card">
          <h2 className="cardTitle">Admin accounts</h2>
          <div className="field">
            <label className="label">Role</label>
            <select
              className="select"
              value={adminRole}
              onChange={(e) => setAdminRole(e.target.value)}
            >
              <option value="institution_admin">Institution admin</option>
              <option value="department_admin">Department admin</option>
              <option value="faculty">Faculty</option>
            </select>
          </div>
          <div className="field">
            <label className="label">Institution</label>
            <select
              className="select"
              value={adminInstId}
              onChange={(e) => setAdminInstId(e.target.value)}
            >
              <option value="">Select institution…</option>
              {institutions.map((i) => (
                <option key={i.institution_id} value={i.institution_id}>
                  {i.name}
                </option>
              ))}
            </select>
          </div>
          {(adminRole === "department_admin" || adminRole === "faculty") && (
            <div className="field">
              <label className="label">Department</label>
              <select
                className="select"
                value={adminDeptId}
                onChange={(e) => setAdminDeptId(e.target.value)}
              >
                <option value="">Select department…</option>
                {departments.map((d) => (
                  <option key={d.department_id} value={d.department_id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="field">
            <label className="label">Username</label>
            <input
              className="input"
              value={adminUsername}
              onChange={(e) => setAdminUsername(e.target.value)}
            />
          </div>
          <div className="field">
            <label className="label">Password (min 8 chars)</label>
            <input
              className="input"
              type="password"
              value={adminPassword}
              onChange={(e) => setAdminPassword(e.target.value)}
            />
          </div>
          <button
            className="btn btnPrimary"
            disabled={
              !adminUsername.trim() ||
              adminPassword.length < 8 ||
              !adminInstId ||
              ((adminRole === "department_admin" || adminRole === "faculty") && !adminDeptId)
            }
            onClick={() =>
              run(
                () =>
                  api.org.createAdminAccount({
                    username: adminUsername.trim(),
                    password: adminPassword,
                    role: adminRole,
                    institution_id: adminInstId || null,
                    department_id:
                      adminRole === "department_admin" || adminRole === "faculty"
                        ? adminDeptId || null
                        : null,
                  }),
                `Admin “${adminUsername.trim()}” created`
              )
            }
          >
            Create admin
          </button>
        </div>

        <div className="card">
          <h2 className="cardTitle">Student Registration Tokens</h2>
          <p className="muted" style={{ marginBottom: "1rem" }}>
            Issue single-use verification tokens for students before they register with their face.
          </p>
          <div className="field">
            <label className="label">Student Roll Number</label>
            <input
              className="input mono"
              value={tokenRollNumber}
              onChange={(e) => setTokenRollNumber(e.target.value)}
              placeholder="e.g. 25BLC1105"
            />
          </div>
          <div className="field">
            <label className="label">Validity (days)</label>
            <input
              type="number"
              min={1}
              max={365}
              className="input"
              value={tokenDays}
              onChange={(e) => setTokenDays(Number(e.target.value))}
              style={{ width: 120 }}
            />
          </div>
          <button
            className="btn btnPrimary"
            disabled={!tokenRollNumber.trim() || issuingToken}
            onClick={async () => {
              setError("");
              setNotice("");
              setIssuingToken(true);
              setTokenCopied(false);
              try {
                const res = await api.admin.issueRegistrationToken(tokenRollNumber.trim(), tokenDays);
                setIssuedToken({ token: res.token, roll: res.roll_number, expires: res.expires_at });
                setNotice(`Registration token generated for ${res.roll_number}`);
                setTokenRollNumber("");
              } catch (e) {
                setError(e instanceof ApiError ? e.message : "Failed to issue token");
              } finally {
                setIssuingToken(false);
              }
            }}
          >
            {issuingToken ? "Generating…" : "Generate token"}
          </button>

          {issuedToken && (
            <div
              style={{
                marginTop: "1.25rem",
                padding: "1rem",
                background: "var(--bg)",
                borderRadius: "8px",
                border: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "0.5rem",
                }}
              >
                <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>
                  Active Token for <span className="mono">{issuedToken.roll}</span>
                </span>
                <span className="badge badgeSuccess">Single-Use</span>
              </div>
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <input
                  readOnly
                  className="input mono"
                  value={issuedToken.token}
                  style={{ background: "var(--surface)", fontWeight: 600 }}
                />
                <button
                  type="button"
                  className="btn btnSecondary"
                  onClick={() => {
                    navigator.clipboard.writeText(issuedToken.token);
                    setTokenCopied(true);
                    setTimeout(() => setTokenCopied(false), 2500);
                  }}
                >
                  {tokenCopied ? "✓ Copied" : "Copy"}
                </button>
              </div>
              <p className="muted" style={{ fontSize: "0.78rem", marginTop: "0.4rem" }}>
                Give this token to the student to enter on the registration page.
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
