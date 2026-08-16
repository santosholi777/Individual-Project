# DeepVisionAttend

A web-based, privacy-conscious student attendance platform built on deep-learning
face recognition. Developed as the BSc (Hons) Computing Individual Project
(ST6001CEM) at Softwarica College of IT & E-Commerce, in academic partnership
with Coventry University.

## Overview

DeepVisionAttend marks attendance from an ordinary camera image. The recognition
engine runs two pre-trained models from the [InsightFace](https://github.com/deepinsight/insightface)
project — **SCRFD** for face detection and an **ArcFace ResNet-50** network that
turns each face into a 512-dimensional embedding. Identity is decided by cosine
similarity against an enrolled gallery, guarded by a configurable confidence
threshold.

A deliberate privacy choice runs through the system: **photographs are never
stored**. Only irreversible embeddings persist, so an enrolled record cannot be
turned back into a recognisable face.

## Repository layout

| Path | Description |
|------|-------------|
| `ai-service/` | Python recognition service (FastAPI), auth, data layer, evaluation harness |
| `frontend/`   | React + TypeScript dashboard (Vite) — kiosk, enrolment, reporting |

See `ai-service/README.md` and `frontend/README.md` for setup and run
instructions for each component.

## Key features

- Contactless attendance capture from a live camera or uploaded image
- Open-set recognition that **rejects unenrolled strangers**, resisting proxy attendance
- Confidence scores and runner-up candidates on every decision for transparency
- JWT authentication with role-based access control
- Repository-based data layer designed for a clean MongoDB migration
- Automated test suite and a reproducible evaluation harness (rank-1, ROC/AUC, EER, d′)

## Evaluation summary

On a labelled subset of Labeled Faces in the Wild (16 enrolled identities, 6
held-out strangers): rank-one accuracy **97.9%**, ROC AUC **0.992**, equal error
rate **1.4%**, and **zero** of 60 stranger images accepted.

## Privacy & data protection

Biometric embeddings, attendance records, model weights and secrets are excluded
from version control. Deleting a student removes their biometric data, not just
metadata. The design applies GDPR data-minimisation and purpose-limitation
principles directly in code.

## Author

Shantosh Oli — Softwarica College of IT & E-Commerce (Coventry University).
