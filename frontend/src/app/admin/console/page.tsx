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
      </div>
    </main>
  );
}
