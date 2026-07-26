/**
 * Transient notifications.
 *
 * Messages are announced in a polite live region — a screen-reader user gets
 * told that a student was registered, not just a silent visual pop.
 *
 * The context and `useToast` hook live in `toast-context.ts`; see the note there.
 */

import { useCallback, useMemo, useState, type ReactNode } from "react";
import { IconAlert, IconCheckCircle, IconInfo, IconX } from "./icons";
import { ToastContext, type ToastContextValue, type ToastTone } from "./toast-context";
import "./Toast.css";

interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  description?: string;
}

const ICONS: Record<ToastTone, ReactNode> = {
  success: <IconCheckCircle />,
  error: <IconAlert />,
  info: <IconInfo />,
};

/** How long each tone stays on screen. Errors linger — they need reading. */
const DURATION: Record<ToastTone, number> = {
  success: 3500,
  info: 4000,
  error: 7000,
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (tone: ToastTone, title: string, description?: string) => {
      const id = Date.now() + Math.random();
      setToasts((current) => [...current, { id, tone, title, description }]);
      window.setTimeout(() => dismiss(id), DURATION[tone]);
    },
    [dismiss],
  );

  const value = useMemo<ToastContextValue>(
    () => ({
      success: (title, description) => push("success", title, description),
      error: (title, description) => push("error", title, description),
      info: (title, description) => push("info", title, description),
    }),
    [push],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast--${toast.tone}`}>
            <span className="toast__icon">{ICONS[toast.tone]}</span>
            <div className="toast__body">
              <p className="toast__title">{toast.title}</p>
              {toast.description && (
                <p className="toast__description">{toast.description}</p>
              )}
            </div>
            <button
              className="toast__close"
              onClick={() => dismiss(toast.id)}
              aria-label="Dismiss notification"
            >
              <IconX size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
