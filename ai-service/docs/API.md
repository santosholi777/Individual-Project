# DeepVisionAttend AI Service — API Reference

Base URL (development): `http://localhost:8000`
Interactive documentation: `http://localhost:8000/docs` (Swagger) · `/redoc`

All request and response bodies are JSON unless stated otherwise. All timestamps
are ISO-8601 **UTC**. All confidence scores are **cosine similarity in `[0, 1]`**.

---

## Contents

| Method | Path | Purpose |
|---|---|---|
| `GET` | [`/health`](#get-health) | Liveness, model and index status |
| `POST` | [`/register`](#post-register) | Enrol a student (file upload) |
| `POST` | [`/register/base64`](#post-registerbase64) | Enrol a student (JSON/base64) |
| `GET` | [`/students`](#get-students) | List registered students |
| `GET` | [`/students/{id}`](#get-studentsid) | Fetch one student |
| `DELETE` | [`/students/{id}`](#delete-studentsid) | Delete a student and their face data |
| `POST` | [`/recognize`](#post-recognize) | Recognise faces (file upload) |
| `POST` | [`/recognize/base64`](#post-recognizebase64) | Recognise faces (JSON/base64) |
| `POST` | [`/attendance`](#post-attendance) | Mark attendance manually |
| `GET` | [`/attendance`](#get-attendance) | Query attendance records |
| `GET` | [`/attendance/summary`](#get-attendancesummary) | Daily present/absent totals |

**Which variant should the frontend use?** The `/base64` endpoints exist for
browsers: `canvas.toDataURL()` output can be posted as-is. The multipart
endpoints suit file pickers, `curl`, and server-to-server calls from Node.

---

## Error format

Every 4xx/5xx raised by this service returns the same shape:

```json
{
  "error_code": "no_face_detected",
  "message": "No face was detected. Ensure the face is clearly visible, well lit and close enough to the camera.",
  "details": { "min_det_score": 0.5, "min_face_size": 50 }
}
```

`error_code` is stable and safe to branch on. `message` is safe to show a user.

| `error_code` | Status | Meaning |
|---|---|---|
| `image_decode_error` | 400 | Payload was not a readable image, or exceeded the size limit |
| `student_not_found` | 404 | Unknown `student_id` |
| `student_already_exists` | 409 | Id already enrolled; pass `overwrite` to replace |
| `duplicate_attendance` | 409 | Already marked for this date/session |
| `empty_gallery` | 409 | Recognition attempted before anyone was registered |
| `no_face_detected` | 422 | No face passed the detector thresholds |
| `multiple_faces_detected` | 422 | More than one face where exactly one is required |
| `low_quality_image` | 422 | Face too blurry, too dark or over-exposed |
| `registration_error` | 422 | Too few usable images, or a blank name |
| `repository_error` | 500 | Storage failure, or an unsafe `student_id` |
| `model_load_error` | 503 | Weights could not be downloaded or initialised |
| `camera_error` | 503 | Camera unavailable (CLI only) |

FastAPI's own request validation (a missing field, a malformed date) returns
`422` with FastAPI's standard `{"detail": [...]}` shape.

---

## `GET /health`

Liveness and readiness. Use for monitoring and container health checks.
`status` is `degraded` when the models are not loaded.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "service": "DeepVisionAttend AI Service",
  "version": "1.0.0",
  "models_ready": true,
  "model_info": {
    "pack": "buffalo_l",
    "downloaded": true,
    "device": "cpu",
    "providers": ["CPUExecutionProvider"]
  },
  "index": {
    "indexed_embeddings": 12,
    "indexed_students": 3,
    "recognition_threshold": 0.45
  },
  "storage": {
    "backend": "local-files",
    "students": 3,
    "attendance_records": 47
  },
  "timestamp": "2026-07-15T14:53:04.904934Z"
}
```

---

## `POST /register`

Enrol a student from uploaded face images. `multipart/form-data`.
Each image must contain **exactly one** clearly visible face.

**Form fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `student_id` | string | yes | Unique id. Letters, digits, `.`, `_`, `-` only; max 64 chars |
| `name` | string | yes | Display name |
| `files` | file[] | yes | Face images (JPEG/PNG/BMP/WEBP), one face each |
| `overwrite` | bool | no | Replace an existing enrolment (default `false`) |

```bash
curl -X POST http://localhost:8000/register \
  -F "student_id=CS2021001" \
  -F "name=Aditi Sharma" \
  -F "files=@face1.jpg" \
  -F "files=@face2.jpg" \
  -F "files=@face3.jpg"
```

**`201 Created`**

```json
{
  "success": true,
  "student": {
    "student_id": "CS2021001",
    "name": "Aditi Sharma",
    "embedding_count": 3,
    "created_at": "2026-07-15T14:53:15.732031Z",
    "updated_at": "2026-07-15T14:53:15.732031Z",
    "metadata": {}
  },
  "accepted_images": 3,
  "rejected_images": 0,
  "rejections": [],
  "total_embeddings": 3
}
```

Images are processed independently: a blurred frame is reported in `rejections`
and skipped rather than failing the request, as long as at least
`DVA_REGISTRATION_MIN_IMAGES` (default 3) survive.

> **Privacy:** the uploaded images are embedded and then discarded. Only the
> 512-D vectors are written to disk.

**Errors:** `409` already registered · `422` too few usable images · `400`
undecodable upload · `500` unsafe `student_id`.

---

## `POST /register/base64`

The browser-friendly variant. Accepts `canvas.toDataURL()` output directly (the
`data:image/...;base64,` prefix is optional).

```bash
curl -X POST http://localhost:8000/register/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "CS2021001",
    "name": "Aditi Sharma",
    "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
    "overwrite": false,
    "metadata": {"department": "Computer Science", "year": "3"}
  }'
```

Response is identical to `POST /register`. Unlike the multipart variant, this
one accepts a `metadata` object stored alongside the student.

```javascript
// React example
const capture = () => canvasRef.current.toDataURL("image/jpeg", 0.92);

await fetch("http://localhost:8000/register/base64", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    student_id: "CS2021001",
    name: "Aditi Sharma",
    images: [capture(), capture(), capture()],
  }),
});
```

---

## `GET /students`

```bash
curl http://localhost:8000/students
```

```json
{
  "count": 1,
  "students": [
    {
      "student_id": "CS2021001",
      "name": "Aditi Sharma",
      "embedding_count": 3,
      "created_at": "2026-07-15T14:53:15.732031Z",
      "updated_at": "2026-07-15T14:53:15.732031Z",
      "metadata": {}
    }
  ]
}
```

---

## `GET /students/{id}`

```bash
curl http://localhost:8000/students/CS2021001
```

Returns the student object. **`404`** with `student_not_found` if unknown.

---

## `DELETE /students/{id}`

Removes the student **and every face embedding held for them**. This is the
consent-withdrawal path.

```bash
curl -X DELETE http://localhost:8000/students/CS2021001
```

```json
{
  "success": true,
  "message": "Student 'CS2021001' and their face embeddings were deleted"
}
```

---

## `POST /recognize`

Identify every face in an image. `multipart/form-data`.

**Form fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | Image containing one or more faces |
| `mark_attendance` | bool | no | Also record attendance for recognised students (default `false`) |
| `session` | string | no | Session label for attendance (default `general`) |
| `max_faces` | int | no | Process at most this many faces, largest first |

```bash
# Recognise only
curl -X POST http://localhost:8000/recognize -F "file=@probe.jpg"

# Kiosk: recognise and mark attendance in one call
curl -X POST http://localhost:8000/recognize \
  -F "file=@probe.jpg" \
  -F "mark_attendance=true" \
  -F "session=lecture-1"
```

**`200 OK`**

```json
{
  "success": true,
  "faces_detected": 1,
  "recognized_count": 1,
  "elapsed_ms": 140.24,
  "results": [
    {
      "recognized": true,
      "student_id": "CS2021001",
      "name": "Aditi Sharma",
      "confidence": 0.9803,
      "margin": 0.9149,
      "bbox": { "x1": 36.6, "y1": 34.88, "x2": 148.38, "y2": 187.96 },
      "det_score": 0.8161,
      "candidates": [
        { "student_id": "CS2021001", "name": "Aditi Sharma", "similarity": 0.9803 },
        { "student_id": "CS2021002", "name": "Rahul Verma", "similarity": 0.0654 }
      ]
    }
  ],
  "attendance": [
    {
      "status": "marked",
      "reason": null,
      "record": {
        "student_id": "CS2021001",
        "name": "Aditi Sharma",
        "timestamp": "2026-07-15T14:53:15.886273Z",
        "date": "2026-07-15",
        "confidence": 0.9803,
        "session": "lecture-1",
        "source": "auto"
      }
    }
  ]
}
```

**Reading the response**

- An **unknown face still returns `200`** with `recognized: false`,
  `student_id: null` and its `candidates` — so the frontend can draw a red
  "Unknown" box, and you can tell a near-miss from a total stranger.
- `margin` is the gap to the runner-up. A high `confidence` with a *low* `margin`
  means two enrolled students look alike — worth surfacing to an admin.
- `attendance` is present only when `mark_attendance=true`, and is index-aligned
  with `results`.

**Errors:** `409 empty_gallery` if nobody is registered · `400` undecodable image.

---

## `POST /recognize/base64`

```bash
curl -X POST http://localhost:8000/recognize/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "mark_attendance": true,
    "session": "lecture-1",
    "max_faces": 1
  }'
```

Response identical to `POST /recognize`.

```javascript
// React example: recognise the current webcam frame
const res = await fetch("http://localhost:8000/recognize/base64", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image: canvasRef.current.toDataURL("image/jpeg", 0.92),
    mark_attendance: true,
    session: "lecture-1",
    max_faces: 1,
  }),
});
const data = await res.json();
const hit = data.results[0];
if (hit?.recognized) {
  console.log(`${hit.name} — ${(hit.confidence * 100).toFixed(1)}%`);
}
```

---

## `POST /attendance`

Mark attendance for a known student **without an image** — manual entry by staff,
or a fallback after a recognition failure.

| Field | Type | Required | Description |
|---|---|---|---|
| `student_id` | string | yes | Must already be registered |
| `confidence` | float | no | `0.0`–`1.0`; use `1.0` for manual entries (default `1.0`) |
| `session` | string | no | Session label (default `general`) |
| `source` | string | no | `manual` (default) or `auto` |

```bash
curl -X POST http://localhost:8000/attendance \
  -H "Content-Type: application/json" \
  -d '{"student_id": "CS2021001", "session": "lecture-1", "source": "manual"}'
```

**`200 OK`**

```json
{
  "status": "marked",
  "reason": null,
  "record": {
    "student_id": "CS2021001",
    "name": "Aditi Sharma",
    "timestamp": "2026-07-15T14:53:15.886273Z",
    "date": "2026-07-15",
    "confidence": 1.0,
    "session": "lecture-1",
    "source": "manual"
  }
}
```

**This endpoint returns `200` in every non-error case.** Check `status`:

| `status` | Meaning |
|---|---|
| `marked` | A new record was written |
| `duplicate` | Already marked for this date/session; `record` holds the original |
| `rejected` | Confidence below `DVA_ATTENDANCE_THRESHOLD`; nothing written |

Duplicates are an *expected* outcome, not a failure: a camera pointed at a
classroom re-recognises the same student many times per minute. `source: manual`
bypasses the confidence gate (staff override), but never the duplicate rule.

**Errors:** `404 student_not_found`.

---

## `GET /attendance`

Query the register. Every filter is optional and they combine with AND.

| Query param | Type | Description |
|---|---|---|
| `student_id` | string | Restrict to one student |
| `date_from` | date | Earliest date, inclusive (`YYYY-MM-DD`) |
| `date_to` | date | Latest date, inclusive (`YYYY-MM-DD`) |
| `session` | string | Restrict to one session label |

```bash
curl "http://localhost:8000/attendance"
curl "http://localhost:8000/attendance?student_id=CS2021001"
curl "http://localhost:8000/attendance?date_from=2026-07-01&date_to=2026-07-15&session=lecture-1"
```

```json
{
  "count": 1,
  "records": [
    {
      "student_id": "CS2021001",
      "name": "Aditi Sharma",
      "timestamp": "2026-07-15T14:53:15.886273Z",
      "date": "2026-07-15",
      "confidence": 0.9803,
      "session": "lecture-1",
      "source": "auto"
    }
  ]
}
```

Records are returned oldest first.

---

## `GET /attendance/summary`

Present/absent totals for one day — the admin dashboard's headline numbers.

| Query param | Type | Description |
|---|---|---|
| `date` | date | Day to summarise (`YYYY-MM-DD`); defaults to today (UTC) |

```bash
curl "http://localhost:8000/attendance/summary?date=2026-07-15"
```

```json
{
  "date": "2026-07-15",
  "total_students": 2,
  "present": 1,
  "absent": 1,
  "attendance_rate": 50.0,
  "records": [ { "student_id": "CS2021001", "...": "..." } ],
  "absentees": [ { "student_id": "CS2021002", "name": "Rahul Verma" } ]
}
```

`absentees` is every registered student with no record on that date.

---

## Typical integration flow

```
1. Enrol once per student
   POST /register/base64   { student_id, name, images: [3-5 frames] }

2. Each class, per frame from the kiosk camera
   POST /recognize/base64  { image, mark_attendance: true, session: "lecture-1" }
       -> results[0].recognized ? show name + confidence : show "Unknown"
       -> attendance[0].status: "marked" | "duplicate" | "rejected"

3. Dashboard
   GET /students
   GET /attendance/summary?date=2026-07-15
   GET /attendance?session=lecture-1

4. Consent withdrawal
   DELETE /students/CS2021001
```

**Polling rate.** CPU inference takes roughly 140 ms for a single face and
~480 ms for six. Sending a frame every 500 ms–1 s is comfortable; the duplicate
rule makes repeat frames harmless, so the frontend does not need to track who has
already been marked.

---

## Notes for the Node.js backend

- This service performs **no authentication**. Do not expose it to the internet.
  Put JWT auth and role-based access control in the Node layer and keep this
  service on a private network.
- It is **stateless over HTTP**: any request can go to any instance, provided
  they share the storage volume (or, later, the MongoDB database).
- CORS origins are configured with `DVA_CORS_ORIGINS`.
- `X-Process-Time` is returned on every response for latency monitoring.
- The OpenAPI schema is available at `/openapi.json` for client generation.
