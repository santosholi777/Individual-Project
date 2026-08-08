# DeepVisionAttend — AI Service

**A Deep Learning–Powered Smart Face Recognition System for Automated Attendance Management in Modern Colleges.**

This is the standalone Python AI microservice: the component that turns a camera
frame into a student identity and an attendance record. It runs on its own, over
HTTP, and is designed to be consumed later by the React frontend and Node.js
backend without any change to the code in this folder.

---

## 1. What it does

| Capability | How |
|---|---|
| **Face detection** | SCRFD-10G (pre-trained, from the InsightFace `buffalo_l` pack) |
| **Face alignment** | Similarity transform onto the ArcFace 112×112 template using 5 landmarks |
| **Face embedding** | **ArcFace R50** (pre-trained on WebFace600K) → 512-D vector |
| **Identity matching** | Cosine similarity against the enrolled gallery |
| **Attendance** | Threshold gate → duplicate prevention → timestamped record |

**No model is trained here.** The system performs inference only, using
pre-trained weights that download automatically on first run (~281 MB).

### Verified accuracy on this build

Measured with the InsightFace sample group photograph (six people), one person
enrolled and the other five acting as impostors:

| Measure | Value |
|---|---|
| Genuine match (same person, re-presented) | **0.98** |
| Impostor scores (five unenrolled people) | −0.08 … **0.07** |
| Separation gap | **0.91** |
| False positives | **0 / 5** |
| Recognition threshold | 0.45 (sits inside the gap, far from both classes) |
| Inference latency (6 faces, CPU, M-series) | ~480 ms |

That gap is why the default threshold is safe: genuine and impostor scores are
not close to each other, so the exact threshold is not delicate.

---

## 2. Requirements

- **Python 3.11 or newer**
- **MongoDB** (accounts only; face data stays on the filesystem)
  - macOS: `brew tap mongodb/brew && brew install mongodb-community@7.0`
  - Start it: `brew services start mongodb-community@7.0`
- A C compiler (`insightface` builds a small extension during install)
  - macOS: `xcode-select --install`
  - Ubuntu/Debian: `sudo apt install build-essential python3-dev`
- A webcam (only for the CLI tools; the API takes uploaded images)
- Internet access **on first run only**, to download the pre-trained weights

---

## 3. Setup

```bash
cd ai-service

# 1. Create a virtual environment with Python 3.11+
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. (Optional) Configure. Defaults are sensible; nothing here is required.
cp .env.example .env

# 4. Create the first account once the service is running (it becomes admin)
#    Or just use the Sign up page in the React app.

# 5. Verify the install — runs 239 tests with mocked models/database
pytest
```

> **If `pip install insightface` fails**, it is almost always a missing compiler
> or a missing Cython. Run `pip install cython numpy` first, then retry. On
> macOS ensure `xcode-select --install` has been run.

> **Tip:** if `python3.11 -m venv` fails with an `ensurepip`/`pyexpat` error
> (a known Homebrew bottle issue on recent macOS), use
> [uv](https://docs.astral.sh/uv/) instead — it ships its own CPython:
> ```bash
> brew install uv && uv venv --python 3.11 .venv && uv pip install -r requirements.txt
> ```

### First run

The first command that touches a model downloads the `buffalo_l` pack into
`models/`. This takes a minute or two and happens exactly once.

---

## 4. Usage

### 4.1 Register a student (webcam)

```bash
python register.py --student-id CS2021001 --name "Aditi Sharma"
```

Captures 5 frames with a live preview and a countdown; change your pose slightly
between captures. Each frame is detected, aligned, quality-checked and embedded.
**The photographs are discarded — only the 512-D vectors are stored.**

Other sources and options:

```bash
# Enrol from a folder of existing photos instead of the camera
python register.py --student-id CS2021001 --name "Aditi Sharma" --from-folder ./photos/aditi

# Replace an existing enrolment
python register.py --student-id CS2021001 --name "Aditi Sharma" --overwrite

# Headless machine (no preview window)
python register.py --student-id CS2021001 --name "Aditi Sharma" --no-preview
```

### 4.2 Live recognition

```bash
# Recognise only — draws boxes, names and confidence
python recognize.py

# Kiosk mode: recognise and mark attendance for a session
python recognize.py --mark --session lecture-1

# Recognise a single image file and exit
python recognize.py --image ./test/group_photo.jpg
```

Green box = recognised, red = unknown. Press `q` or `ESC` to quit.

### 4.3 Measure the accuracy (for your report)

```bash
python evaluate.py --dataset ./dataset --report evaluation_report.md
```

This is the command that produces your results chapter. See
[section 9](#9-evaluation) for the dataset layout and what it measures.

### 4.4 Attendance reports

```bash
python attendance.py list                          # today
python attendance.py list --student-id CS2021001   # one student
python attendance.py list --from 2026-07-01 --to 2026-07-15
python attendance.py summary                       # present/absent totals
python attendance.py export --output report.csv    # CSV for the project report
python attendance.py mark --student-id CS2021001   # manual entry
```

### 4.5 Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
# or simply:  python app.py
```

Then open **http://localhost:8000/docs** for interactive Swagger documentation.

Full endpoint reference with example requests: [`docs/API.md`](docs/API.md).

---

## 5. Project structure

```
ai-service/
├── app.py                      # FastAPI application (HTTP layer only)
├── config.py                   # Typed, env-driven settings
├── domain.py                   # Framework-free core types
├── schemas.py                  # Pydantic request/response models (API contract)
├── exceptions.py               # Exception hierarchy (carries HTTP status + code)
├── dependencies.py             # Composition root — the ONLY place naming concretes
├── logging_config.py           # Shared logging setup
├── database.py                 # Repository interfaces + local-file implementations
│
├── register.py                 # CLI: webcam enrolment
├── recognize.py                # CLI: live recognition
├── attendance.py               # CLI: reporting and export
├── evaluate.py                 # CLI: accuracy measurement → report
│
├── auth/
│   ├── security.py             # bcrypt hashing, JWT signing, reset tokens
│   ├── models.py               # User / reset-request domain types
│   ├── repository.py           # MongoDB user + reset-token storage
│   ├── service.py              # Signup, login, password-reset rules
│   ├── dependencies.py         # Route guards (get_current_user, require_admin)
│   └── router.py               # The /auth endpoints
│
├── services/
│   ├── engine.py               # Pre-trained model pack management
│   ├── detector.py             # SCRFD face detection
│   ├── embedder.py             # ArcFace alignment + embedding
│   ├── matcher.py              # Cosine similarity gallery index
│   ├── registration_service.py # Enrolment pipeline
│   ├── recognition_service.py  # Pipeline orchestration
│   └── attendance_service.py   # Attendance policy
│
├── evaluation/
│   ├── dataset.py              # Dataset loading + enrol/probe split
│   ├── runner.py               # In-memory gallery, probe scoring
│   ├── metrics.py              # Verification + open-set identification
│   ├── figures.py              # SVG figures for the report
│   └── report.py               # Console / Markdown / JSON / CSV output
│
├── utils/
│   ├── camera.py               # Webcam context manager
│   ├── image_utils.py          # Decoding, quality checks, annotation
│   └── similarity.py           # Cosine similarity maths
│
├── tests/                      # 239 tests; models + database mocked
├── storage/
│   ├── embeddings/             # students.json + vectors/<id>.npy
│   ├── attendance/             # attendance.jsonl
│   └── logs/
├── models/                     # Downloaded pre-trained weights (gitignored)
└── docs/API.md                 # API reference and example requests
```

---

## 6. Architecture

### Layering

```
      HTTP (app.py)          CLI (register/recognize/attendance.py)
              \                /
               \              /
             services/  ← all the intelligence lives here
                   |
             database.py (abstract repositories)
                   |
        local files today  →  MongoDB later
```

The rule that makes this work: **dependencies point inwards, and only
`dependencies.py` names a concrete class.** Services depend on
`StudentRepository`, not `JsonStudentRepository`; on `FaceDetectorProtocol`, not
`InsightFaceDetector`. Nothing in `services/` imports FastAPI, so the same code
serves both the API and the CLIs with no duplicated logic.

### SOLID in practice

| Principle | Where |
|---|---|
| **S**ingle responsibility | `detector` finds faces; `embedder` vectorises them; `matcher` decides identity; `attendance_service` owns policy. Each has one reason to change. |
| **O**pen/closed | A new storage backend or a new face model is a new class, not an edit to an existing one. |
| **L**iskov | Any `StudentRepository` is substitutable — the test suite swaps in fakes and the pipeline cannot tell. |
| **I**nterface segregation | Three narrow repositories rather than one fat `Database`; `FaceDetectorProtocol` and `FaceEmbedderProtocol` are separate. |
| **D**ependency inversion | High-level `RecognitionService` depends on Protocols; concretes are injected by `dependencies.py`. |

### Swapping in MongoDB later

Implement the three interfaces from `database.py` (`StudentRepository`,
`EmbeddingRepository`, `AttendanceRepository`) against Motor/PyMongo, then change
one method — `ServiceContainer._build_database()` in `dependencies.py`. **No
other file changes.** That seam is the reason the repositories are abstract.

---

## 7. Configuration

Every setting is overridable via environment variable (prefix `DVA_`) or `.env`.
See [`.env.example`](.env.example) for the annotated list. The ones that matter:

| Variable | Default | Meaning |
|---|---|---|
| `DVA_RECOGNITION_THRESHOLD` | `0.45` | Minimum cosine similarity to accept an identity |
| `DVA_ATTENDANCE_THRESHOLD` | `0.55` | Minimum similarity to *record* attendance (kept stricter) |
| `DVA_CTX_ID` | `-1` | `-1` = CPU, `0` = first CUDA GPU |
| `DVA_DET_SIZE` | `640,640` | Detector resolution; `320,320` is ~2× faster, misses small faces |
| `DVA_MODEL_PACK` | `buffalo_l` | `buffalo_s` is smaller/faster for weak hardware |
| `DVA_ONCE_PER_SESSION_PER_DAY` | `true` | One attendance record per student per session per day |

**Tuning the threshold.** Raising it reduces false positives (wrong student
marked) at the cost of false negatives (student must retry). Since a wrong name
on the register is worse than a retry, `attendance_threshold` is deliberately
stricter than `recognition_threshold`; the config refuses to start if you invert
that relationship.

### GPU

```bash
pip uninstall onnxruntime && pip install onnxruntime-gpu
export DVA_CTX_ID=0
```

---

## 8. Privacy and ethics

The project's ethical commitments are enforced in code, not just documented:

- **No photographs are stored.** Images are embedded and discarded; only the
  512-D vectors persist. An embedding cannot be viewed as a face. There is a
  test asserting no image file is ever written (`test_stores_no_images_only_embeddings`).
- **Consent is reversible.** `DELETE /students/{id}` removes the student *and*
  their biometric vectors; the repository can erase their attendance history too.
- **Biometric data is never committed.** `.gitignore` excludes
  `storage/embeddings/` and the attendance log.
- **Identifiers are validated.** Student ids are restricted to safe characters, so
  a crafted id cannot escape the storage directory (`../../` is rejected).
- **Decisions are auditable.** Every attendance record carries its confidence
  score and whether it was `auto` or `manual`; every recognition returns runner-up
  candidates so a near-miss is distinguishable from a stranger.

Under GDPR, face embeddings are personal data. Before deploying to real students,
obtain explicit informed consent, serve over HTTPS, and put access control in
front of this service (planned for the Node.js backend layer).

---

## 9. Evaluation

`evaluate.py` measures accuracy against a labelled dataset and writes a report
you can put straight into the project write-up. It uses the **production**
detector, embedder and matcher — an evaluation that measured a different code
path from the deployed one would measure nothing useful — and builds its gallery
in memory, so running it never touches the live registry or attendance log.

### Dataset layout

One folder per student, named by their id:

```
dataset/
├── CS2021001/
│   ├── 01.jpg  02.jpg  03.jpg     # first N enrol, the rest become probes
│   ├── dim_light/    *.jpg        # optional: condition sub-folders
│   ├── side_angle/   *.jpg
│   └── motion_blur/  *.jpg
├── CS2021002/
│   └── ...
└── meta.json                      # optional: names and fairness groups
```

```json
{
  "CS2021001": {"name": "Aditi Sharma", "group": "female"},
  "CS2021002": {"name": "Rahul Verma", "group": "male"}
}
```

Condition sub-folders are what let the report answer *"where does it break?"*
rather than only *"how accurate is it?"* — the exact gap this project's
literature review identifies in prior work.

### Running it

```bash
# Basic
python evaluate.py --dataset ./dataset

# Full run: hold two people out as strangers, write report + figures
python evaluate.py --dataset ./dataset \
    --holdout CS2021005,CS2021006 \
    --report evaluation_report.md \
    --json results.json \
    --csv scores.csv
```

**`--holdout` is the important flag.** Those people are never enrolled, and
every image of them must be rejected. That is the direct measurement of whether
proxy attendance is possible — without it, the false-accept rate against
strangers is simply unmeasured.

### What it reports

| Metric | Question it answers |
|---|---|
| **Rank-1 accuracy** | Is the closest match the right person? (ignores the threshold) |
| ROC AUC, EER, d′ | How separable are "same person" and "different people"? |
| **Correct / Wrong ID / Rejected** | What the attendance system actually does at a threshold |
| **Strangers accepted** | Can an unenrolled person be marked present? |
| By condition | Where does it break — dim light, angle, blur? |
| By group | Fairness sanity check (with an honest caveat about sample size) |

The three operational outcomes are deliberately kept apart, because they are not
equally serious: a **wrong ID** puts the wrong name on the register silently; a
**rejection** just asks the student to look again.

Outputs: a console summary, a Markdown report, two SVG figures (score
distribution and threshold trade-off), plus JSON and per-probe CSV for
re-analysis.

### The threshold recommendation

`evaluate.py` also suggests a threshold. It does **not** pick the strictest value
that scores perfectly — that would sit exactly on the lowest genuine score in
your sample and reject the next slightly-worse face. It takes the midpoint of the
range that scores best, which is the max-margin choice and generalises.

---

## 10. Testing

```bash
pytest                          # 239 tests, ~9 seconds, no model download or database
pytest -v                       # verbose
pytest tests/test_api.py        # HTTP contract only
pytest tests/test_evaluation.py # evaluation metrics only
pytest tests/test_auth.py       # accounts, login, password reset
```

The suite replaces the two ONNX networks with deterministic fakes, so it runs
offline and fast. Everything else — quality gates, storage, threshold logic,
duplicate prevention, error mapping, evaluation metrics — is the real production
code path.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModelLoadError: could not download` | No internet on first run. Connect once; weights are then cached in `models/`. |
| `CameraError: could not open camera 0` | Another app holds the camera, or macOS camera permission is not granted to your terminal (System Settings → Privacy & Security → Camera). |
| `pip install insightface` fails | Missing compiler/Cython. `pip install cython numpy`, then `xcode-select --install` (macOS) or `apt install build-essential python3-dev`. |
| `empty_gallery` (409) from `/recognize` | No students registered yet. Run `register.py` first. |
| Recognition is slow | Set `DVA_DET_SIZE=320,320`, or use `DVA_MODEL_PACK=buffalo_s`, or enable the GPU. |
| Student not recognised | Re-enrol with more varied poses/lighting (`register.py --overwrite`), or lower `DVA_RECOGNITION_THRESHOLD` slightly. |
| Wrong student recognised | **Raise** `DVA_RECOGNITION_THRESHOLD`. Check for near-identical enrolments (e.g. twins). |

---

## 12. Next integration steps

This service is deliberately standalone and stateless over HTTP. To integrate:

1. **Node.js backend** → call `POST /recognize/base64` and `GET /attendance`;
   add JWT auth and role-based access control at that layer.
2. **React frontend** → capture with `canvas.toDataURL()` and post to the
   `/base64` endpoints; the CORS origins are already configurable.
3. **MongoDB** → implement the three repository interfaces, change one method in
   `dependencies.py`.

---

## 13. References

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular
  Margin Loss for Deep Face Recognition.* CVPR, 4690–4699.
- Guo, J. et al. (2021). *Sample and Computation Redistribution for Efficient
  Face Detection (SCRFD).* — the detector in the `buffalo_l` pack.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified
  Embedding for Face Recognition and Clustering.* CVPR, 815–823.
- InsightFace project: https://github.com/deepinsight/insightface
