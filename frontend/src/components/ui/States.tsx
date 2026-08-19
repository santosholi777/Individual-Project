/**
 * Loading, empty and error states.
 *
 * Centralised because these three are what a user actually sees most of the
 * time, and an app that treats them as afterthoughts feels broken. Each error
 * here explains the cause *and* the next action.
 */

import type { ReactNode } from "react";
import { ApiError } from "../../api/client";
import { Button } from "./Button";
import { IconAlert, IconRefresh } from "./icons";
import "./States.css";

/** Indeterminate spinner. */
export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <span
      className="spinner"
      style={{ width: size, height: size }}
      role="status"
      aria-label="Loading"
    />
  );
}

/** Full-panel loading state. */
export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state">
      <Spinner size={24} />
      <p className="state__title">{label}</p>
    </div>
  );
}

/** Grey placeholder blocks that mirror the shape of the content being loaded. */
export function Skeleton({
  height = 16,
  width = "100%",
  radius = "var(--radius-sm)",
}: {
  height?: number | string;
  width?: number | string;
  radius?: string;
}) {
  return (
    <span
      className="skeleton"
      style={{ height, width, borderRadius: radius }}
      aria-hidden="true"
    />
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="state">
      {icon && <div className="state__icon">{icon}</div>}
      <p className="state__title">{title}</p>
      {description && <p className="state__description">{description}</p>}
      {action && <div className="state__action">{action}</div>}
    </div>
  );
}

/**
 * Render an ApiError with its cause and a retry.
 *
 * The offline case gets its own copy because "cannot reach the AI service" has a
 * completely different fix from "the service refused this request".
 */
export function ErrorState({
  error,
  onRetry,
}: {
  error: ApiError;
  onRetry?: () => void;
}) {
  return (
    <div className="state">
      <div className="state__icon state__icon--error">
        <IconAlert size={22} />
      </div>
      <p className="state__title">
        {error.isOffline ? "AI service unreachable" : "Something went wrong"}
      </p>
      <p className="state__description">{error.message}</p>
      {!error.isOffline && (
        <code className="state__code">{error.code}</code>
      )}
      {onRetry && (
        <div className="state__action">
          <Button icon={<IconRefresh size={15} />} onClick={onRetry}>
            Try again
          </Button>
        </div>
      )}
    </div>
  );
}
