/**
 * Kiosk: the live camera that recognises students and marks attendance.
 *
 * Design notes that matter here:
 *
 * - Frames are sent on an interval, and a new request is never started while one
 *   is still in flight. CPU inference takes ~150ms; without that guard a slow
 *   response would queue requests until the service drowned.
 * - Face boxes are drawn from the coordinates the *service* returns, scaled from
 *   the captured frame's pixel space into the displayed video's CSS pixels.
 * - The preview is mirrored (a mirror is what people expect of a front camera),
 *   so the overlay is mirrored to match — but the frame sent for recognition is
 *   never mirrored, because a flipped face is a different face to the model.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { AttendanceRecord, RecognitionResult } from "../api/types";
import { PageHeader } from "../components/layout/PageHeader";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import {
  IconAlert,
  IconCamera,
  IconCheckCircle,
  IconScan,
  IconUserPlus,
  IconX,
} from "../components/ui/icons";
import { EmptyState, Spinner } from "../components/ui/States";
import { useToast } from "../components/ui/toast-context";
import { useCamera } from "../hooks/useCamera";
import { percent, timeAgo } from "../utils/format";
import "./Kiosk.css";

/** How often a frame is sent for recognition while scanning. */
const SCAN_INTERVAL_MS = 700;

interface MarkedEntry {
  record: AttendanceRecord;
  at: number;
}

export function Kiosk() {
  const toast = useToast();
  const camera = useCamera(true);

  const [scanning, setScanning] = useState(false);
  const [session, setSession] = useState("lecture-1");
  const [results, setResults] = useState<RecognitionResult[]>([]);
  const [marked, setMarked] = useState<MarkedEntry[]>([]);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);

  const videoBoxRef = useRef<HTMLDivElement>(null);
  const inFlight = useRef(false);
  // Read inside the interval callback, which must not be re-created per render.
  const sessionRef = useRef(session);
  sessionRef.current = session;

  const scanOnce = useCallback(async () => {
    if (inFlight.current) return;

    const frame = camera.capture(0.85);
    if (!frame) return;

    inFlight.current = true;
    try {
      const response = await api.recognize({
        image: frame,
        mark_attendance: true,
        session: sessionRef.current,
      });

      setResults(response.results);
      setLatencyMs(response.elapsed_ms);
      setApiError(null);

      for (const outcome of response.attendance) {
        if (outcome.status === "marked" && outcome.record) {
          const record = outcome.record;
          setMarked((current) => [{ record, at: Date.now() }, ...current]);
          toast.success(
            `${record.name} marked present`,
            `${record.student_id} · ${percent(record.confidence)} confidence`,
          );
        }
      }
    } catch (cause) {
      if (cause instanceof ApiError) {
        setApiError(cause);
        // An empty gallery or a dead service will repeat every tick; stop
        // scanning rather than spamming the screen with the same failure.
        if (cause.code === "empty_gallery" || cause.isOffline) {
          setScanning(false);
        }
      }
    } finally {
      inFlight.current = false;
    }
  }, [camera, toast]);

  useEffect(() => {
    if (!scanning || camera.status !== "ready") return;
    const id = window.setInterval(() => void scanOnce(), SCAN_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [scanning, camera.status, scanOnce]);

  // Boxes are meaningless once scanning stops — they would freeze on screen.
  useEffect(() => {
    if (!scanning) setResults([]);
  }, [scanning]);

  /** Map the service's frame-space box onto the displayed video element. */
  const boxStyle = (result: RecognitionResult) => {
    const box = videoBoxRef.current?.getBoundingClientRect();
    const { width: frameW, height: frameH } = camera.dimensions;
    if (!box || !frameW || !frameH) return undefined;

    const scaleX = box.width / frameW;
    const scaleY = box.height / frameH;
    return {
      left: `${result.bbox.x1 * scaleX}px`,
      top: `${result.bbox.y1 * scaleY}px`,
      width: `${(result.bbox.x2 - result.bbox.x1) * scaleX}px`,
      height: `${(result.bbox.y2 - result.bbox.y1) * scaleY}px`,
    };
  };

  const recognised = results.filter((result) => result.recognized);

  return (
    <div className="page">
      <PageHeader
        title="Take Attendance"
        description="Students look at the camera. Recognised faces are marked present automatically — nobody is marked twice."
      />

      <div className="kiosk">
        <div className="card kiosk__stage">
          <div className="kiosk__video-wrap" ref={videoBoxRef}>
            <video
              ref={camera.videoRef}
              className="kiosk__video"
              playsInline
              muted
              autoPlay
            />

            {/* Overlay is mirrored to match the mirrored preview. */}
            <div className="kiosk__overlay">
              {results.map((result, index) => (
                <div
                  key={index}
                  className={`face ${result.recognized ? "face--known" : "face--unknown"}`}
                  style={boxStyle(result)}
                >
                  <span className="face__tag">
                    {result.recognized
                      ? `${result.name} · ${percent(result.confidence)}`
                      : "Unknown"}
                  </span>
                </div>
              ))}
            </div>

            {camera.status !== "ready" && (
              <div className="kiosk__camera-state">
                {camera.status === "starting" ? (
                  <>
                    <Spinner size={26} />
                    <p>Starting camera…</p>
                  </>
                ) : (
                  <>
                    <IconCamera size={30} />
                    <p className="kiosk__camera-error">
                      {camera.error ?? "Camera is off"}
                    </p>
                    <Button onClick={() => void camera.start()}>Enable camera</Button>
                  </>
                )}
              </div>
            )}

            {scanning && camera.status === "ready" && (
              <div className="kiosk__scanline" aria-hidden="true" />
            )}

            <div className="kiosk__hud">
              <Badge tone={scanning ? "good" : "neutral"}>
                {scanning ? "● Scanning" : "Paused"}
              </Badge>
              {latencyMs !== null && scanning && (
                <Badge tone="neutral">{Math.round(latencyMs)} ms</Badge>
              )}
              {scanning && (
                <Badge tone={recognised.length ? "accent" : "neutral"}>
                  {results.length} face{results.length === 1 ? "" : "s"}
                </Badge>
              )}
            </div>
          </div>

          <div className="kiosk__controls">
            <label className="field field--inline">
              <span className="field__label">Session</span>
              <input
                className="input"
                value={session}
                onChange={(event) => setSession(event.target.value)}
                placeholder="lecture-1"
                disabled={scanning}
              />
            </label>

            {camera.devices.length > 1 && (
              <label className="field field--inline">
                <span className="field__label">Camera</span>
                <select
                  className="input"
                  value={camera.deviceId ?? ""}
                  onChange={(event) => camera.selectDevice(event.target.value)}
                  disabled={scanning}
                >
                  {camera.devices.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              </label>
            )}

            <div className="spacer" />

            <Button
              variant={scanning ? "danger" : "primary"}
              size="lg"
              icon={scanning ? <IconX size={16} /> : <IconScan size={16} />}
              onClick={() => setScanning((value) => !value)}
              disabled={camera.status !== "ready"}
            >
              {scanning ? "Stop scanning" : "Start scanning"}
            </Button>
          </div>

          {apiError && (
            <div className="kiosk__banner" role="alert">
              <IconAlert size={16} />
              <div>
                <strong>{apiError.message}</strong>
                {apiError.code === "empty_gallery" && (
                  <p>
                    <Link to="/register">Register a student</Link> before scanning.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        <aside className="card kiosk__side">
          <div className="kiosk__side-head">
            <h2 className="kiosk__side-title">Marked present</h2>
            <Badge tone={marked.length ? "good" : "neutral"}>{marked.length}</Badge>
          </div>

          {marked.length === 0 ? (
            <EmptyState
              icon={<IconCheckCircle size={22} />}
              title="Nobody yet"
              description={
                scanning
                  ? "Waiting for a registered face to appear."
                  : "Press Start scanning to begin the session."
              }
            />
          ) : (
            <ul className="marked">
              {marked.map((entry) => (
                <li key={`${entry.record.student_id}-${entry.at}`} className="marked__item">
                  <span className="marked__check" aria-hidden="true">
                    <IconCheckCircle size={16} />
                  </span>
                  <div className="marked__text">
                    <span className="marked__name">{entry.record.name}</span>
                    <span className="marked__meta">
                      {entry.record.student_id} · {percent(entry.record.confidence)}
                    </span>
                  </div>
                  <span className="marked__time">{timeAgo(entry.record.timestamp)}</span>
                </li>
              ))}
            </ul>
          )}

          <div className="kiosk__side-foot">
            <Link to="/register" className="unstyled-link">
              <Button fullWidth icon={<IconUserPlus size={15} />}>
                Register a new student
              </Button>
            </Link>
          </div>
        </aside>
      </div>
    </div>
  );
}
