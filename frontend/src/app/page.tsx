import Link from "next/link";

export default function Home() {
  return (
    <main className="page">
      <section className="hero">
        <h1 className="heroTitle">Attendance that takes itself.</h1>
        <p className="heroTagline">
          One classroom photo marks the register. Face matching is always scoped to
          your section&apos;s roster (~60-70 students) — fast and fair whether the
          campus has 500 or 50,000 students. Uncertain faces go to a human review
          queue, never a forced guess.
        </p>
        <div className="heroActions">
          <Link className="btn btnPrimary" href="/register">
            Register with face
          </Link>
          <Link className="btn btnSecondary" href="/student/login">
            Check my attendance
          </Link>
          <Link className="btn btnSecondary" href="/admin/login">
            Admin login
          </Link>
        </div>
      </section>

      <div className="grid2">
        <div className="card">
          <h2 className="cardTitle">For students</h2>
          <p className="muted">
            Register once with 3-5 quick camera shots, then just attend class.
            Check your history any time.
          </p>
        </div>
        <div className="card">
          <h2 className="cardTitle">For faculty</h2>
          <p className="muted">
            Upload one photo at the start of class — results arrive in seconds.
            Review any uncertain matches yourself.
          </p>
        </div>
      </div>
    </main>
  );
}
