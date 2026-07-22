/**
 * Webcam access for the registration and kiosk pages.
 *
 * Wraps getUserMedia so that permission handling, track cleanup and frame
 * capture live in one place. Forgetting to stop the tracks leaves the camera
 * light on after navigation, so teardown is handled here rather than per page.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus = "idle" | "starting" | "ready" | "denied" | "error";

export interface CameraState {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  error: string | null;
  devices: MediaDeviceInfo[];
  deviceId: string | null;
  selectDevice: (id: string) => void;
  start: () => Promise<void>;
  stop: () => void;
  /** Grab the current frame as a JPEG data URI, or null if not ready. */
  capture: (quality?: number) => string | null;
  /** Natural pixel size of the video feed — needed to scale face boxes. */
  dimensions: { width: number; height: number };
}

/**
 * Open a webcam and expose frame capture.
 *
 * @param active When false, the camera is stopped and released.
 */
export function useCamera(active: boolean = true): CameraState {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("error");
      setError(
        "This browser cannot access cameras. Chrome, Edge and Safari all support it over http://localhost.",
      );
      return;
    }

    setStatus("starting");
    setError(null);

    try {
      streamRef.current?.getTracks().forEach((track) => track.stop());

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {
          // Autoplay can reject if the element is remounted mid-play; the
          // stream is still attached, so this is not fatal.
        });
      }

      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings();
      setDimensions({
        width: settings?.width ?? 1280,
        height: settings?.height ?? 720,
      });

      // Labels are only populated once permission has been granted, so the
      // device list is enumerated after getUserMedia rather than before.
      const all = await navigator.mediaDevices.enumerateDevices();
      const cameras = all.filter((device) => device.kind === "videoinput");
      setDevices(cameras);
      if (!deviceId && cameras[0]) setDeviceId(cameras[0].deviceId);

      setStatus("ready");
    } catch (cause) {
      const name = (cause as DOMException)?.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        setStatus("denied");
        setError(
          "Camera permission was denied. Allow camera access for this site in your browser, then try again.",
        );
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        setStatus("error");
        setError("No camera was found. Connect a webcam and try again.");
      } else if (name === "NotReadableError") {
        setStatus("error");
        setError(
          "The camera is already in use by another application. Close it and try again.",
        );
      } else {
        setStatus("error");
        setError(`Could not start the camera: ${String(cause)}`);
      }
    }
  }, [deviceId]);

  const capture = useCallback((quality = 0.92): string | null => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return null;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) return null;

    // The preview is mirrored for the user's comfort, but the frame sent to the
    // model must not be — a mirrored face is a different face to the embedder.
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
  }, []);

  const selectDevice = useCallback((id: string) => setDeviceId(id), []);

  useEffect(() => {
    if (active) {
      void start();
    } else {
      stop();
    }
    return stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, deviceId]);

  return {
    videoRef,
    status,
    error,
    devices,
    deviceId,
    selectDevice,
    start,
    stop,
    capture,
    dimensions,
  };
}
