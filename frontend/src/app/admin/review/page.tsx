"use client";

/**
 * /admin/review — resolve unmatched faces.
 *
 * Default path: assign from the SESSION SECTION's roster (fast, scoped).
 * Edge case: the explicit "Search all students" button triggers the slower,
 * institution-scoped vector search — never used automatically.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import api, { ApiError, tokenStore } from "@/lib/api";
import { absoluteUrl } from "@/lib/api";
import type { RosterStudent, ReviewItem, SearchAllCandidate, Section } from "@/lib/types";

type AssignState = {
  reviewId: string;
  candidates: SearchAllCandidate[];
} | null;

export default function ReviewQueuePage() {
  const router = useRouter();
  const [sections, setSections] = useState<Section[]>([]);
  const [sectionFilter, setSectionFilter] = useState("");
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [roster, setRoster] = useState<RosterStudent[]>([]);
  const [assignFor, setAssignFor] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<AssignState>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const loadQueue = useCallback(
    async (sectionId: string) => {
      setLoading(true);
      setError("");
      try {
        setItems(await api.admin.listReviewQueue(sectionId || undefined));
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) {
          tokenStore.clear();
          router.replace("/admin/login");
        } else {
          setError(e instanceof ApiError ? e.message : "Failed to load review queue");
        }
      } finally {
        setLoading(false);
      }
    },
    [router]
  );

  useEffect(() => {
    if (!tokenStore.access || tokenStore.role === "student") {
      router.replace("/admin/login");
      return;
    }
    api.org
      .listSections()
      .then((rows) => setSections(rows))
      .catch(() => undefined);
    void loadQueue("");
  }, [router, loadQueue]);

  const openAssign = async (review: ReviewItem) => {
    setAssignFor(review.review_id);
    setSearchResults(null);
    try {
      setRoster(await api.admin.listStudentsInSection(review.section_id));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load roster");
    }
  };

  const assignTo = async (reviewId: string, studentId: string | null, searchAll: boolean) => {
    setBusyId(reviewId);
    setError("");
    try {
      await api.admin.resolveReview(reviewId, studentId ? "assign" : "discard", studentId ?? undefined, searchAll);
      setItems((prev) => prev.filter((i) => i.review_id !== reviewId));
      setAssignFor(null);
      setSearchResults(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to resolve item");
    } finally {
      setBusyId(null);
    }
  };

  const runSearchAll = async (reviewId: string) => {
    setBusyId(reviewId);
    setError("");
    try {
      const res = await api.admin.searchAllForReview(reviewId);
      setSearchResults({ reviewId, candidates: res.candidates });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Institution-wide search failed");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <main className="page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 className="pageTitle">Review queue</h1>
          <p className="pageSubtitle">
            Faces that didn&apos;t match confidently. Assign them yourself or discard.
          </p>
        </div>
        <div className="field" style={{ minWidth: 220, marginBottom: 0 }}>
          <select
            className="select"
            value={sectionFilter}
            onChange={(e) => {
              setSectionFilter(e.target.value);
              void loadQueue(e.target.value);
            }}
          >
            <option value="">All my sections</option>
            {sections.map((s) => (
              <option key={s.section_id} value={s.section_id}>
                {s.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <div className="alert alertError">{error}</div>}
      {loading && <div className="alert alertInfo">Loading…</div>}

      {!loading && items.length === 0 && (
        <div className="card">
          <p className="empty">Nothing pending — all faces are resolved.</p>
        </div>
      )}

      {items.map((item) => (
        <div className="card" key={item.review_id}>
          <div className="reviewCard">
            {item.face_crop_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                className="reviewCrop"
                src={absoluteUrl(item.face_crop_url) ?? ""}
                alt="Unmatched face crop"
              />
            ) : (
              <div className="reviewCrop" />
            )}
            <div style={{ flex: 1, minWidth: 240 }}>
              <div className="btnRow">
                <span className="badge badgeInfo">{item.section_name}</span>
                <span className="mono muted">{new Date(item.created_at).toLocaleString()}</span>
              </div>
              <p className="muted" style={{ margin: "0.4rem 0" }}>
                Closest roster guess:{" "}
                {item.best_guess_distance != null ? (
                  <>
                    distance <span className="mono">{item.best_guess_distance.toFixed(3)}</span>{" "}
                    — above threshold, so not auto-marked.
                  </>
                ) : (
                  "no roster candidate found."
                )}
              </p>

              {assignFor !== item.review_id && (
                <div className="btnRow">
                  <button className="btn btnPrimary btnSm" onClick={() => openAssign(item)}>
                    Assign student…
                  </button>
                  <button
                    className="btn btnDanger btnSm"
                    disabled={busyId === item.review_id}
                    onClick={() => assignTo(item.review_id, null, false)}
                  >
                    Discard
                  </button>
                  <button
                    className="btn btnSecondary btnSm"
                    disabled={busyId === item.review_id}
                    onClick={() => runSearchAll(item.review_id)}
                  >
                    {busyId === item.review_id ? "Searching…" : "Search all students"}
                  </button>
                </div>
              )}

              {searchResults?.reviewId === item.review_id && (
                <div style={{ marginTop: "0.6rem" }}>
                  <div className="alert alertInfo">
                    Institution-wide search results (slower fallback — pick carefully):
                  </div>
                  {searchResults.candidates.length === 0 ? (
                    <p className="muted">No candidates found in this institution.</p>
                  ) : (
                    searchResults.candidates.map((c) => (
                      <div className="btnRow" key={c.student_id} style={{ marginBottom: "0.35rem" }}>
                        <span>
                          {c.name} · <span className="mono">{c.roll_number}</span>{" "}
                          · distance {c.distance.toFixed(3)}
                        </span>
                        <button
                          className="btn btnSuccess btnSm"
                          disabled={busyId === item.review_id}
                          onClick={() => assignTo(item.review_id, c.student_id, true)}
                        >
                          Assign
                        </button>
                      </div>
                    ))
                  )}
                </div>
              )}

              {assignFor === item.review_id && (
                <div style={{ marginTop: "0.6rem" }}>
                  <div className="alert alertInfo">
                    Pick a student from this section&apos;s roster:
                  </div>
                  {roster.length === 0 ? (
                    <p className="muted">No enrolled students in this section.</p>
                  ) : (
                    <select
                      className="select"
                      defaultValue=""
                      onChange={(e) => e.target.value && assignTo(item.review_id, e.target.value, false)}
                    >
                      <option value="">Select student…</option>
                      {roster.map((s) => (
                        <option key={s.student_id} value={s.student_id}>
                          {s.roll_number} — {s.name}
                        </option>
                      ))}
                    </select>
                  )}
                  <div className="btnRow" style={{ marginTop: "0.5rem" }}>
                    <button className="btn btnSecondary btnSm" onClick={() => setAssignFor(null)}>
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </main>
  );
}
