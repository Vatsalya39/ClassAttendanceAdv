"use client";

/**
 * src/components/CameraCapture.tsx
 * ────────────────────────────────
 * Guided camera capture used by BOTH student registration (3-5 face shots)
 * and the faculty attendance dashboard (single classroom photo).
 *
 * - getUserMedia works only on HTTPS or localhost (browser rule); anything
 *   else degrades gracefully to a file-upload fallback.
 *
 * Black-screen fix: the MediaStream must be attached to the <video> element
 * AFTER React has rendered it. We therefore store the stream in state and
 * assign `video.srcObject` inside an effect — assigning it directly inside
 * the click handler races against the re-render and yields a black frame.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface PoseStep {
  id: string;
  label: string;
  instruction: string;
  badge?: string;
}

interface CameraCaptureProps {
  shots: Blob[];
  onChange: (shots: Blob[]) => void;
  minShots: number;
  maxShots: number;
  /** What is being captured — used in helper copy, e.g. "face shots". */
  subject?: string;
  /**
   * Minimum acceptable SHORTER-side resolution in px. Captures below the
   * floor are blocked with a warning (session photos need ~1024px+ for
   * back-row faces to be recognisable; enrollment close-ups don't).
   */
  minResolution?: number;
  /** Optional sequence of guided poses for multi-angle registration. */
  poseSteps?: PoseStep[];
}

type CameraState = "idle" | "starting" | "live" | "denied" | "unavailable" | "insecure";

/** Request full-HD+ so back-row faces stay recognisable in classroom shots. */
const CAMERA_CONSTRAINTS: MediaTrackConstraints = {
  width: { ideal: 1920 },
  height: { ideal: 1080 },
  facingMode: "user",
};

export default function CameraCapture({
  shots,
  onChange,
  minShots,
  maxShots,
  subject = "photos",
  minResolution = 0,
  poseSteps,
}: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CameraState>("idle");
  const [errorHint, setErrorHint] = useState<string>("");
  const [thumbnails, setThumbnails] = useState<string[]>([]);
  const [fileFallbackNames, setFileFallbackNames] = useState<string[]>([]);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setState("idle");
  }, []);

  // Release the camera when the component unmounts.
  useEffect(() => stopStream, [stopStream]);

  // Keep thumbnail object-URLs in sync with the shots array.
  useEffect(() => {
    const urls = shots.map((b) => URL.createObjectURL(b));
    setThumbnails(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [shots]);

  /**
   * THE FIX: once state === "live" React has rendered the <video>, so the
   * stream can safely be attached. Doing this in the click handler attached
   * it to nothing (videoRef.current was still null) → black screen.
   */
  useEffect(() => {
    const video = videoRef.current;
    if (state !== "live" || !video || !streamRef.current) return;
    video.srcObject = streamRef.current;
    const playing = video.play();
    if (playing !== undefined) playing.catch(() => undefined);
  }, [state]);

  const startCamera = async () => {
    setErrorHint("");
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setState("insecure");
      setErrorHint(
        "This browser cannot access a camera here (needs HTTPS or localhost). Upload photos instead below."
      );
      return;
    }
    setState("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: CAMERA_CONSTRAINTS,
        audio: false,
      });
      streamRef.current = stream;
      setState("live"); // effect above now attaches the stream to <video>
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError") {
        setState("denied");
        setErrorHint(
          "Camera permission was denied. Allow camera access in your browser settings, or upload photos instead below."
        );
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        setState("unavailable");
        setErrorHint("No camera was found on this device. Upload photos instead below.");
      } else {
        setState("insecure");
        setErrorHint(
          "Camera requires HTTPS or localhost. Upload photos instead below."
        );
      }
    }
  };

  const captureShot = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    // Resolution floor: block captures that the backend would reject anyway
    // (session photos need enough pixels for back-row faces).
    if (minResolution > 0 && Math.min(video.videoWidth, video.videoHeight) < minResolution) {
      setErrorHint(
        `Camera resolution is ${video.videoWidth}x${video.videoHeight}px — at least ` +
          `${minResolution}px on the shorter side is required. Close other ` +
          "camera apps and retry, or upload a high-resolution photo instead."
      );
      return;
    }
    setErrorHint("");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        if (shots.length >= maxShots) return;
        onChange([...shots, blob]);
        if (shots.length + 1 >= maxShots) stopStream();
      },
      "image/jpeg",
      0.95
    );
  };

  const removeShot = (idx: number) => {
    onChange(shots.filter((_, i) => i !== idx));
  };

  const onFilesPicked = (files: FileList | null) => {
    if (!files) return;
    const remaining = maxShots - shots.length;
    const picked = Array.from(files).slice(0, Math.max(0, remaining));
    setFileFallbackNames(picked.map((f) => f.name));
    onChange([...shots, ...picked]);
  };

  const canCaptureMore = shots.length < maxShots;
  const isLive = state === "live";

  const numPoseSteps = poseSteps?.length ?? 0;
  const activePoseIndex = numPoseSteps > 0 && shots.length < numPoseSteps ? shots.length : -1;
  const currentPose = activePoseIndex >= 0 && poseSteps ? poseSteps[activePoseIndex] : null;
  const allPosesCompleted = numPoseSteps > 0 && shots.length >= numPoseSteps;

  const helperCopy =
    maxShots === 1
      ? `Click one clear ${subject} — look at the preview, then press Capture.`
      : poseSteps
      ? `Follow the guided angles below to capture high-accuracy reference templates.`
      : `Take ${minShots}-${maxShots} ${subject} in good light. Look straight at the camera and keep your face inside the frame.`;

  return (
    <div>
      {poseSteps && (
        <div className="poseGuideContainer">
          <div className="poseStepper">
            {poseSteps.map((step, idx) => {
              const isDone = idx < shots.length;
              const isActive = idx === shots.length;
              return (
                <span
                  key={step.id}
                  className={`poseStepItem ${
                    isDone ? "poseStepItemDone" : isActive ? "poseStepItemActive" : ""
                  }`}
                >
                  {isDone ? "✓ " : isActive ? "● " : "○ "}
                  {step.label}
                </span>
              );
            })}
          </div>
          {currentPose ? (
            <p className="poseInstruction">
              <strong>Pose {activePoseIndex + 1} of {poseSteps.length} ({currentPose.label}):</strong>{" "}
              {currentPose.instruction}
            </p>
          ) : allPosesCompleted ? (
            <p className="poseInstruction" style={{ color: "var(--success)" }}>
              ✓ All required pose angles captured! You can add up to {maxShots - shots.length} optional shot
              {maxShots - shots.length === 1 ? "" : "s"} or continue to finalize registration.
            </p>
          ) : null}
        </div>
      )}

      <div className="cameraBox">
        {/* Always rendered so the ref exists BEFORE the stream attaches. */}
        <video
          ref={videoRef}
          className="cameraVideo"
          style={{ visibility: isLive ? "visible" : "hidden" }}
          autoPlay
          playsInline
          muted
        />
        {/* Face alignment reticle and pose banner when pose guidance is active */}
        {isLive && poseSteps && !allPosesCompleted && (
          <div className="cameraFaceReticle" aria-hidden="true" />
        )}
        {isLive && currentPose && (
          <div className="cameraPoseBanner">
            {currentPose.label}: {currentPose.instruction}
          </div>
        )}
        {!isLive && (
          <div className="cameraPlaceholder">
            {state === "idle" && "Camera is off — press “Start camera” to begin."}
            {state === "starting" && "Starting camera…"}
            {state === "denied" && "Camera blocked — you can still upload photos."}
            {state === "unavailable" && "No camera found — upload photos below."}
            {state === "insecure" && "Camera needs HTTPS or localhost — upload below."}
          </div>
        )}
      </div>

      <div className="btnRow" style={{ marginTop: "0.75rem" }}>
        {!isLive && (
          <button
            type="button"
            className="btn btnSecondary"
            onClick={startCamera}
            disabled={!canCaptureMore || state === "starting"}
          >
            {state === "starting" ? "Starting…" : "Start camera"}
          </button>
        )}
        {isLive && (
          <>
            <button
              type="button"
              className="btn btnPrimary"
              onClick={captureShot}
              disabled={!canCaptureMore}
            >
              {currentPose
                ? `Capture ${currentPose.label} (${shots.length + 1}/${maxShots})`
                : `Capture (${shots.length}/${maxShots})`}
            </button>
            <button type="button" className="btn btnSecondary" onClick={stopStream}>
              Stop camera
            </button>
          </>
        )}
        <label className="btn btnSecondary">
          Upload instead
          <input
            type="file"
            accept="image/jpeg,image/png"
            multiple={maxShots > 1}
            hidden
            onChange={(e) => onFilesPicked(e.target.files)}
          />
        </label>
      </div>

      {errorHint && <div className="alert alertWarning">{errorHint}</div>}
      {!errorHint && !isLive && shots.length < maxShots && (
        <p className="muted" style={{ marginTop: "0.5rem" }}>
          {helperCopy}
        </p>
      )}
      {!errorHint && isLive && (
        <p className="muted" style={{ marginTop: "0.5rem" }}>
          Live preview active — keep your face inside the dashed guide and press Capture when ready.
        </p>
      )}

      {shots.length > 0 && (
        <div className="shotStrip">
          {shots.map((_, idx) =>
            fileFallbackNames[idx] && !thumbnails[idx]?.startsWith("blob:") ? null : (
              <div className="shotItem" key={idx}>
                <div className="shotThumb">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={thumbnails[idx]} alt={`Captured ${idx + 1}`} />
                  <button
                    type="button"
                    className="shotRemove"
                    aria-label={`Remove capture ${idx + 1}`}
                    onClick={() => removeShot(idx)}
                  >
                    ×
                  </button>
                </div>
                {poseSteps && (
                  <span className="shotPoseTag">
                    {poseSteps[idx]?.badge ?? poseSteps[idx]?.label ?? `Shot ${idx + 1}`}
                  </span>
                )}
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}
