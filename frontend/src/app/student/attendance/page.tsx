"use client";

/**
 * /student/attendance — the logged-in student's attendance history.
 * Redirects to login when no student token is present.
 */

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError, tokenStore } from "@/lib/api";
import type { StudentAttendanceResponse } from "@/lib/types";

export default function MyAttendancePage() {
  const router = useRouter();
  const [data, setData] = useState<StudentAttendanceResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!tokenStore.access || tokenStore.role !== "student") {
      router.replace("/student/login");
      return;
    }
    api.students
      .myAttendance()
      .then(setData)
      .catch((e: ApiError) => {
        if (e.status === 401) {
          tokenStore.clear();
          router.replace("/student/login");
        } else {
          setError(e.message);
        }
      })
      .finally(() => setLoading(false));
  }, [router]);

  const logout = () => {
    tokenStore.clear();
    router.push("/");
  };

  const present = data?.records.length ?? 0;

  return (
    <main className="page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 className="pageTitle">My attendance</h1>
          <p className="pageSubtitle">Every session where your face was matched</p>
        </div>
        <button className="btn btnSecondary btnSm" onClick={logout}>
          Log out
        </button>
      </div>

      {loading && <div className="alert alertInfo">Loading…</div>}
      {error && <div className="alert alertError">{error}</div>}

      {data && (
        <div className="card">
          <div className="statRow" style={{ marginTop: 0, marginBottom: "1rem" }}>
            <div className="stat">
              <span className="statValue">{present}</span>
              <span className="statLabel">Sessions attended</span>
            </div>
          </div>
          {data.records.length === 0 ? (
            <p className="empty">No attendance records yet — see you in class!</p>
          ) : (
            <div className="tableWrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Course</th>
                    <th>Marked at</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.records.map((r) => (
                    <tr key={r.session_id}>
                      <td>{r.session_date}</td>
                      <td>{r.course_name}</td>
                      <td className="muted">
                        {r.marked_at ? new Date(r.marked_at).toLocaleTimeString() : "—"}
                      </td>
                      <td>
                        <span className="badge badgeSuccess">Present</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
