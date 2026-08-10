/**
 * Student enrolment: details → consent → capture → done.
 *
 * The consent step is not decoration. Face embeddings are biometric personal
 * data; the project's compliance requirements call for explicit informed consent
 * before enrolment, so the capture button is genuinely gated behind it.
 *
 * The AI service rejects blurred or badly lit frames individually and reports
 * why, so those reasons are surfaced verbatim rather than collapsed into a
 * generic failure.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { PageHeader } from "../components/layout/PageHeader";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import {
  IconAlert,
  IconCamera,
  IconCheck,
  IconCheckCircle,
  IconChevronLeft,
  IconShield,
  IconUserPlus,
  IconX,
} from "../components/ui/icons";
import { Spinner } from "../components/ui/States";
import { useToast } from "../components/ui/toast-context";
import { useCamera } from "../hooks/useCamera";
import { useMutation } from "../hooks/useApi";
import "./Register.css";

/** How many frames to collect. The service needs at least 3 usable ones. */
const TARGET_SHOTS = 5;

const POSE_HINTS = [
  "Look straight at the camera",
  "Turn your head slightly left",
  "Turn your head slightly right",
  "Tilt your chin up a little",
  "Smile naturally",
];

type Step = "details" | "consent" | "capture" | "done";

export function Register() {
  const toast = useToast();
  const navigate = useNavigate();

  const [step, setStep] = useState<Step>("details");
  const [studentId, setStudentId] = useState("");
  const [name, setName] = useState("");
  const [department, setDepartment] = useState("");
  const [consented, setConsented] = useState(false);
  const [shots, setShots] = useState<string[]>([]);
  const [overwrite, setOverwrite] = useState(false);
  const [flash, setFlash] = useState(false);
  const [result, setResult] = useState<{
    accepted: number;
    rejected: number;
    rejections: string[];
  } | null>(null);

  const camera = useCamera(step === "capture");
  const register = useMutation(api.register);

  // The id names a file on the service's disk, so it validates the same set.
  const idValid = /^[A-Za-z0-9._-]{1,64}$/.test(studentId.trim());
  const detailsValid = idValid && name.trim().length > 0;

  const takeShot = () => {
    const frame = camera.capture();
    if (!frame) {
      toast.error("Could not capture the frame", "Is the camera still running?");
      return;
    }
    setShots((current) => [...current, frame]);
    setFlash(true);
    window.setTimeout(() => setFlash(false), 180);
  };

  const submit = async () => {
    try {
      const response = await register.run({
        student_id: studentId.trim(),
        name: name.trim(),
        images: shots,
        overwrite,
        metadata: department.trim() ? { department: department.trim() } : {},
      });

      setResult({
        accepted: response.accepted_images,
        rejected: response.rejected_images,
        rejections: response.rejections,
      });
      setStep("done");
      camera.stop();
      toast.success(
        `${response.student.name} registered`,
        `${response.total_embeddings} face embeddings stored`,
      );
    } catch (cause) {
      if (cause instanceof ApiError && cause.code === "student_already_exists") {
        toast.error(
          "That student ID is already registered",
          "Tick 'Replace existing enrolment' to overwrite them.",
        );
        setOverwrite(true);
      } else if (cause instanceof ApiError) {
        toast.error("Registration failed", cause.message);
      }
    }
  };

  const restart = () => {
    setStep("details");
    setStudentId("");
    setName("");
    setDepartment("");
    setConsented(false);
    setShots([]);
    setOverwrite(false);
    setResult(null);
    register.reset();
  };

  return (
    <div className="page">
      <PageHeader
        title="Register Student"
        description="Capture a student's face so the system can recognise them. Photos are never stored — only the mathematical representation."
      />

      <Steps current={step} />

      <div className="reg">
        {/* ---------------------------------------------------- Details */}
        {step === "details" && (
          <div className="card reg__panel">
            <h2 className="reg__title">Student details</h2>
            <p className="reg__sub">These identify the student on the register.</p>

            <div className="reg__form">
              <label className="field">
                <span className="field__label">
                  Student ID <span className="field__req">*</span>
                </span>
                <input
                  className="input"
                  value={studentId}
                  onChange={(event) => setStudentId(event.target.value)}
                  placeholder="CS2021001"
                  autoFocus
                />
                <span className={`field__hint ${studentId && !idValid ? "field__hint--error" : ""}`}>
                  {studentId && !idValid
                    ? "Use only letters, numbers, dot, underscore or hyphen."
                    : "The college roll number."}
                </span>
              </label>

              <label className="field">
                <span className="field__label">
                  Full name <span className="field__req">*</span>
                </span>
                <input
                  className="input"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Aditi Sharma"
                />
                <span className="field__hint">Shown on the attendance register.</span>
              </label>

              <label className="field">
                <span className="field__label">Department</span>
                <input
                  className="input"
                  value={department}
                  onChange={(event) => setDepartment(event.target.value)}
                  placeholder="Computer Science"
                />
                <span className="field__hint">Optional.</span>
              </label>
            </div>

            <div className="reg__actions">
              <Button
                variant="primary"
                disabled={!detailsValid}
                onClick={() => setStep("consent")}
              >
                Continue
              </Button>
            </div>
          </div>
        )}

        {/* ---------------------------------------------------- Consent */}
        {step === "consent" && (
          <div className="card reg__panel">
            <div className="reg__consent-head">
              <span className="reg__consent-icon">
                <IconShield size={20} />
              </span>
              <div>
                <h2 className="reg__title">Consent to biometric enrolment</h2>
                <p className="reg__sub">
                  Read this to <strong>{name}</strong> before capturing their face.
                </p>
              </div>
            </div>

            <ul className="reg__terms">
              <li>
                <IconCheck size={15} />
                <span>
                  Your face will be converted into a list of numbers (a
                  “embedding”). <strong>No photograph is kept.</strong> The images
                  are deleted the moment they are converted.
                </span>
              </li>
              <li>
                <IconCheck size={15} />
                <span>
                  The numbers cannot be turned back into a picture of your face.
                </span>
              </li>
              <li>
                <IconCheck size={15} />
                <span>
                  The data is used <strong>only</strong> to mark your attendance.
                </span>
              </li>
              <li>
                <IconCheck size={15} />
                <span>
                  You can withdraw at any time. Deleting your record erases your
                  face data permanently.
                </span>
              </li>
            </ul>

            <label className="reg__checkbox">
              <input
                type="checkbox"
                checked={consented}
                onChange={(event) => setConsented(event.target.checked)}
              />
              <span>
                The student has been informed and gives their consent to enrol.
              </span>
            </label>

            <div className="reg__actions">
              <Button icon={<IconChevronLeft size={15} />} onClick={() => setStep("details")}>
                Back
              </Button>
              <div className="spacer" />
              <Button
                variant="primary"
                disabled={!consented}
                icon={<IconCamera size={15} />}
                onClick={() => setStep("capture")}
              >
                Start capture
              </Button>
            </div>
          </div>
        )}

        {/* ---------------------------------------------------- Capture */}
        {step === "capture" && (
          <div className="reg__capture">
            <div className="card reg__panel">
              <div className="reg__video-wrap">
                <video
                  ref={camera.videoRef}
                  className="reg__video"
                  playsInline
                  muted
                  autoPlay
                />
                <div className="reg__guide" aria-hidden="true" />
                {flash && <div className="reg__flash" aria-hidden="true" />}

                {camera.status !== "ready" && (
                  <div className="reg__camera-state">
                    {camera.status === "starting" ? (
                      <>
                        <Spinner size={24} />
                        <p>Starting camera…</p>
                      </>
                    ) : (
                      <>
                        <IconCamera size={28} />
                        <p>{camera.error ?? "Camera is off"}</p>
                        <Button onClick={() => void camera.start()}>Enable camera</Button>
                      </>
                    )}
                  </div>
                )}

                <div className="reg__hint">
                  {shots.length < TARGET_SHOTS
                    ? POSE_HINTS[shots.length]
                    : "All shots captured"}
                </div>
              </div>

              <div className="reg__capture-bar">
                <div className="reg__progress">
                  <span className="reg__progress-text">
                    {shots.length} of {TARGET_SHOTS} captured
                  </span>
                  <div className="reg__progress-track">
                    <div
                      className="reg__progress-fill"
                      style={{ width: `${(shots.length / TARGET_SHOTS) * 100}%` }}
                    />
                  </div>
                </div>

                <Button
                  variant="primary"
                  size="lg"
                  icon={<IconCamera size={16} />}
                  onClick={takeShot}
                  disabled={camera.status !== "ready" || shots.length >= TARGET_SHOTS}
                >
                  Capture
                </Button>
              </div>
            </div>

            <div className="card reg__panel reg__review">
              <h2 className="reg__title">Captured shots</h2>
              <p className="reg__sub">
                Take {TARGET_SHOTS} shots, changing your pose slightly each time.
                More variety means more reliable recognition.
              </p>

              <div className="reg__thumbs">
                {Array.from({ length: TARGET_SHOTS }, (_, index) => (
                  <div
                    key={index}
                    className={`thumb ${shots[index] ? "thumb--filled" : ""}`}
                  >
                    {shots[index] ? (
                      <>
                        <img src={shots[index]} alt={`Capture ${index + 1}`} />
                        <button
                          className="thumb__remove"
                          onClick={() =>
                            setShots((current) =>
                              current.filter((_, position) => position !== index),
                            )
                          }
                          aria-label={`Remove capture ${index + 1}`}
                        >
                          <IconX size={12} />
                        </button>
                      </>
                    ) : (
                      <span className="thumb__index">{index + 1}</span>
                    )}
                  </div>
                ))}
              </div>

              <label className="reg__checkbox reg__checkbox--compact">
                <input
                  type="checkbox"
                  checked={overwrite}
                  onChange={(event) => setOverwrite(event.target.checked)}
                />
                <span>Replace existing enrolment for this ID</span>
              </label>

              {register.error && (
                <div className="reg__error" role="alert">
                  <IconAlert size={15} />
                  <div>
                    <strong>{register.error.message}</strong>
                    {Array.isArray(register.error.details.rejections) && (
                      <ul>
                        {(register.error.details.rejections as string[]).map((reason) => (
                          <li key={reason}>{reason}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              )}

              <div className="reg__actions">
                <Button onClick={() => setStep("consent")} disabled={register.loading}>
                  Back
                </Button>
                <div className="spacer" />
                <Button
                  variant="primary"
                  loading={register.loading}
                  disabled={shots.length < 3}
                  icon={<IconUserPlus size={15} />}
                  onClick={() => void submit()}
                >
                  {register.loading ? "Processing…" : "Register student"}
                </Button>
              </div>
              {shots.length < 3 && (
                <p className="reg__note">At least 3 usable shots are required.</p>
              )}
            </div>
          </div>
        )}

        {/* ------------------------------------------------------- Done */}
        {step === "done" && result && (
          <div className="card reg__panel reg__done">
            <span className="reg__done-icon">
              <IconCheckCircle size={30} />
            </span>
            <h2 className="reg__title">{name} is registered</h2>
            <p className="reg__sub">
              Their face is now recognised by the system. The photographs have
              been discarded.
            </p>

            <div className="reg__summary">
              <div className="reg__summary-item">
                <span className="reg__summary-value">{result.accepted}</span>
                <span className="reg__summary-label">shots accepted</span>
              </div>
              <div className="reg__summary-item">
                <span className="reg__summary-value">{result.rejected}</span>
                <span className="reg__summary-label">shots rejected</span>
              </div>
              <div className="reg__summary-item">
                <Badge tone="good" icon={<IconCheck />}>
                  Embeddings stored
                </Badge>
              </div>
            </div>

            {result.rejections.length > 0 && (
              <div className="reg__rejections">
                <p className="reg__rejections-title">Rejected shots</p>
                <ul>
                  {result.rejections.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="reg__actions reg__actions--center">
              <Button onClick={restart} icon={<IconUserPlus size={15} />}>
                Register another
              </Button>
              <Button variant="primary" onClick={() => navigate("/students")}>
                View students
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/** The wizard's progress rail. */
function Steps({ current }: { current: Step }) {
  const steps: { key: Step; label: string }[] = [
    { key: "details", label: "Details" },
    { key: "consent", label: "Consent" },
    { key: "capture", label: "Capture" },
    { key: "done", label: "Done" },
  ];
  const index = steps.findIndex((step) => step.key === current);

  return (
    <ol className="steps">
      {steps.map((step, position) => (
        <li
          key={step.key}
          className={`steps__item ${position < index ? "steps__item--done" : ""} ${
            position === index ? "steps__item--current" : ""
          }`}
        >
          <span className="steps__marker">
            {position < index ? <IconCheck size={12} /> : position + 1}
          </span>
          <span className="steps__label">{step.label}</span>
        </li>
      ))}
    </ol>
  );
}
