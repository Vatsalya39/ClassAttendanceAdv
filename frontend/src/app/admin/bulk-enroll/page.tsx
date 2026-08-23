"use client";

/**
 * /admin/bulk-enroll — upload a CSV of roll numbers against a section.
 */

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError, tokenStore } from "@/lib/api";
import type { BulkEnrollResult, Section } from "@/lib/types";

export default function BulkEnrollPage() {
  const router = useRouter();
  const [sections, setSections] = useState<Section[]>([]);
  const [sectionId, setSectionId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<BulkEnrollResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement | null>(null);

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

  const submit = async () => {
    if (!sectionId || !file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(await api.admin.bulkEnrollCsv(sectionId, file));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Bulk enroll failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="page pageNarrow">
      <h1 className="pageTitle">Bulk enroll</h1>
      <p className="pageSubtitle">
        Upload a single-column CSV of roll numbers to enroll them into a section.
      </p>

      <div className="card">
        <div className="field">
          <label className="label">Section</label>
          <select
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
          <label className="label">CSV file (one roll number per line)</label>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,text/csv"
            className="input"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

        {error && <div className="alert alertError">{error}</div>}
        {result && (
          <div className="alert alertSuccess">
            Enrolled {result.enrolled} student{result.enrolled === 1 ? "" : "s"}
            {result.skipped > 0 && ` · skipped ${result.skipped} (already enrolled or not found)`}.
          </div>
        )}

        <button className="btn btnPrimary" disabled={!sectionId || !file || busy} onClick={submit}>
          {busy ? "Uploading…" : "Enroll students"}
        </button>
      </div>
    </main>
  );
}
