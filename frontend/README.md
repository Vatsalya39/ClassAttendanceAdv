# Frontend

Next.js 14 (App Router) frontend for the multi-campus attendance system.

## Pages

| Route | Purpose |
|---|---|
| `/` | Home — register / student / admin entry points |
| `/register` | Student enrollment: details + guided camera capture (3-5 shots) → `POST /register` |
| `/student/login`, `/student/attendance` | Student JWT login + attendance history |
| `/admin/login` | Admin login (JWT with role + scope claims) |
| `/admin/dashboard` | Section picker, upload classroom photo, **poll** capture job (queued → processing → done), matched vs needs-review results |
| `/admin/review` | Review queue: assign from section roster (default), discard, or the explicit slower "search all students" |
| `/admin/reports` | Attendance report with section + date filters |
| `/admin/bulk-enroll` | CSV of roll numbers → `POST /admin/sections/{id}/bulk-enroll` |
| `/admin/console` | Super-admin: institutions, departments, sections, admin accounts |

## Development

```bash
npm install
npm run dev        # http://localhost:3000
```

The API base URL comes from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## Notes

- All HTTP calls live in **one module**: `src/lib/api.ts`. No page imports
  `fetch` directly.
- Camera access requires HTTPS or localhost (browser rule). Permission denial
  and missing cameras degrade gracefully to a file-upload fallback.
- Tokens are kept in localStorage and sent as Bearer headers; moving to
  httpOnly cookies is a Phase 5 hardening step isolated to `api.ts`.
