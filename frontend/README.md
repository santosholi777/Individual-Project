# DeepVisionAttend — Web Frontend

The React interface for **DeepVisionAttend**: register students, take attendance
from a live camera, and review the register.

This is the presentation layer only. All intelligence lives in the Python AI
service (`../ai-service`), which this app talks to over HTTP.

---

## 1. Screens

| Route | Screen | What it does |
|---|---|---|
| `/` | **Dashboard** | Today's attendance rate, present/absent totals, a 7-day trend, recent check-ins and the absentee list |
| `/kiosk` | **Take Attendance** | Live camera. Recognises faces and marks attendance automatically, with a running "marked present" feed |
| `/register` | **Register Student** | Four-step wizard: details → **consent** → capture 5 shots → done |
| `/students` | **Students** | The enrolled register: search, inspect face-data counts, delete |
| `/attendance` | **Attendance Log** | Filter by date range, session or student; export CSV |

---

## 2. Setup

**Requirements:** Node.js 20+ and the AI service running.

```bash
# 1. Start the AI service first (in another terminal)
cd ../ai-service
.venv/bin/python -m uvicorn app:app        # http://localhost:8000

# 2. Then the frontend
cd ../frontend
npm install
npm run dev                                # http://localhost:5173
```

Open **http://localhost:5173**.

> **The camera only works on `localhost` or HTTPS.** Browsers block
> `getUserMedia` on other insecure origins. This is a browser rule, not an app
> bug — if you deploy this, it must be behind HTTPS.

### Configuration

Point the app at a different AI service by creating `.env`:

```bash
VITE_API_URL=http://localhost:8000
```

If you change the port the frontend runs on, add that origin to
`DVA_CORS_ORIGINS` in `../ai-service/.env`, or the browser will block the calls.

### Commands

```bash
npm run dev       # dev server with hot reload
npm run build     # typecheck + production build into dist/
npm run preview   # serve the production build
npm run lint      # oxlint
```

---

## 3. Structure

```
frontend/src/
├── api/
│   ├── client.ts          # The ONLY place that calls fetch. Typed, one error shape.
│   └── types.ts           # TypeScript mirror of the service's contract
├── components/
│   ├── layout/            # AppShell (nav + live service status), PageHeader
│   ├── charts/            # StatTile, AttendanceChart
│   └── ui/                # Button, Badge, Modal, Toast, States, icons
├── hooks/
│   ├── useApi.ts          # useQuery / useMutation
│   ├── useCamera.ts       # getUserMedia, device list, frame capture
│   └── useTheme.ts        # light / dark / system
├── pages/                 # One file per route, with its own CSS
├── styles/
│   ├── tokens.css         # Every colour, space and radius — light + dark
│   └── global.css         # Reset, base elements, utilities
└── utils/format.ts        # UTC → local time, percentages, dates
```

**The rule:** components never call `fetch` and never hold a raw URL. Everything
goes through `api/client.ts`, so the AI service's structured errors
(`{ error_code, message, details }`) become a typed `ApiError` exactly once.

---

## 4. Notes on the design

**Dependencies.** React, React Router and nothing else. No component library, no
data-fetching library, no icon package — the design system, the icons and the two
data hooks are all in this repo. Fewer moving parts to explain in a viva, and
nothing to break on an upgrade.

**Theming.** Every colour is a token in `styles/tokens.css`. Dark mode is a
*selected* set of values, not an inverted light mode, and it follows the OS by
default while letting the in-app toggle win either way.

**The chart.** One column chart, one hue — days are magnitudes, not identities,
so a categorical palette would be wrong. The palette was checked against the
actual light and dark surfaces for contrast and colour-blind separation. Only the
peak column is labelled (a number on every bar goes unread); the rest is carried
by the axis, the hover tooltip and the **View as table** toggle — so nothing is
gated behind colour or a mouse.

**Accessibility.** Real `<table>` markup with captions and scoped headers, a
visible focus ring everywhere, `aria-live` toast announcements, dialogs that trap
Escape and lock scroll, and status always paired with text — never colour alone.

**Kiosk polling.** A frame is sent every 700 ms, and a new request is never
started while one is in flight — otherwise slow CPU inference would queue
requests until the service drowned. Duplicate marking is the *service's* job, so
the UI never has to track who is already present.

**The mirror detail.** The preview is mirrored, because that is what people
expect of a front-facing camera. The frame sent for recognition is **not** —
a flipped face is a different face to the model. The overlay is mirrored to match
the preview, and each label un-mirrored so the text reads correctly.

---

## 5. Verified

Checked against the real AI service with real face data:

- All five screens render, in light and dark, at desktop and mobile widths.
- The full browser → API → recognition → attendance round trip: a real face was
  recognised at **99.5% confidence** and attendance was marked, in ~366 ms.
- Error paths: service offline, camera denied, duplicate ID, empty gallery — each
  shows what happened and what to do about it.
- `npm run build` and `npm run lint` are clean.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| "AI service unreachable" | Start it: `uvicorn app:app` in `../ai-service`. The sidebar dot shows live status. |
| Calls fail but the service is up | CORS. The browser treats `localhost` and `127.0.0.1` as **different origins** — use whichever is in `DVA_CORS_ORIGINS`. |
| "Camera permission was denied" | Allow camera access for the site, and use `localhost`, not a LAN IP. |
| Camera is black / in use | Another app holds it (Zoom, Photo Booth). Close it and press Enable camera. |
| "No students are registered yet" | Register someone first — recognition needs a gallery. |
| Student not recognised | Re-register with more varied poses and better light, or lower `DVA_RECOGNITION_THRESHOLD`. |

---

## 7. Next step: the Node.js backend

This app currently talks **straight to the Python service**, which has no
authentication. That is fine for a local demo and is not fine for real students.

The planned Node/Express layer sits between them and owns:

- **Login and JWT** — the AI service has no concept of users.
- **Role-based access** — a lecturer sees their class; an admin sees everything.
- **MongoDB** — replacing the AI service's local-file storage.

When it exists, point `VITE_API_URL` at Node instead of Python and keep the AI
service on a private network. `api/client.ts` is the only file that needs to
change.
