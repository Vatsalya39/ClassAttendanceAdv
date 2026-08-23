"use client";

/**
 * /admin/reports — attendance report with section + date filters.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError, tokenStore } from "@/lib/api";
import type { AttendanceReport, Section } from "@/lib/types";

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function ReportsPage() {
  const router = useRouter();
  const [sections, setSections] = useState<Section[]>([]);
  const [sectionId, setSectionId] = useState("");
  const [date, setDate] = useState(todayIso());
  const [report, setReport] = useState<AttendanceReport | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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
      .catch(() => undefined);
  }, [router]);

  const load = useCallback(async () => {
    if (!sectionId || !date) return;
    setLoading(true);
    setError("");
    try {
      setReport(await api.admin.getAttendanceReport(sectionId, date));
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        tokenStore.clear();
        router.replace("/admin/login");
      } else {
        setError(e instanceof ApiError ? e.message : "Failed to load report");
      }
    } finally {
      setLoading(false);
    }
  }, [sectionId, date, router]);

  return (
    <main className="page">
      <h1 className="pageTitle">Attendance reports</h1>
      <p className="pageSubtitle">Filter by section and date</p>

      <div className="card">
        <div className="btnRow" style={{ alignItems: "flex-end" }}>
          <div className="field" style={{ marginBottom: 0, minWidth: 220 }}>
            <label className="label">Section</label>
            <select
              className="select"
              value={sectionId}
              onChange={(e) => setSectionId(e.target.value)}
            >
              <option value="">Select a section…</option>
              {sections.map((s) => (
                <option key={s.section_id} value={s.section_id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ marginBottom: 0 }}>
            <label className="label">Date</label>
            <input
              type="date"
              className="input"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
          </div>
          <button className="btn btnPrimary" disabled={!sectionId || !date} onClick={load}>
            Load report
          </button>
        </div>
      </div>

      {loading && <div className="alert alertInfo">Loading…</div>}
      {error && <div className="alert alertError">{error}</div>}

      {report && (
        <div className="card">
          <div className="statRow">
            <div className="stat">
              <span className="statValue">{report.present}</span>
              <span className="statLabel">Present</span>
            </div>
            <div className="stat">
              <span className="statValue">{report.total_enrolled - report.present}</span>
              <span className="statLabel">Not marked</span>
            </div>
            <div className="stat">
              <span className="statValue">{report.total_enrolled}</span>
              <span className="statLabel">Enrolled</span>
            </div>
          </div>
          <div className="tableWrap" style={{ marginTop: "1rem" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Roll number</th>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Marked at</th>
                </tr>
              </thead>
              <tbody>
                {report.rows.map((r) => (
                  <tr key={r.student_id}>
                    <td className="mono">{r.roll_number}</td>
                    <td>{r.name}</td>
                    <td>
                      <span className={`badge ${r.marked ? "badgeSuccess" : "badgeWarning"}`}>
                        {r.marked ? "Present" : "Absent"}
                      </span>
                    </td>
                    <td className="muted">
                      {r.marked_at ? new Date(r.marked_at).toLocaleTimeString() : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </main>
  );
}
