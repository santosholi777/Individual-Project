/**
 * Form pieces shared by the auth pages.
 *
 * Split out because a login field and a signup field differing subtly is how
 * inconsistent forms happen.
 */

import { useState, type InputHTMLAttributes, type ReactNode } from "react";
import { IconAlert } from "../../components/ui/icons";
import "./AuthForm.css";

export interface AuthFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  /** Shown under the field in red; also marks the input as invalid. */
  error?: string | null;
  hint?: string;
}

export function AuthField({
  label,
  error,
  hint,
  id,
  className = "",
  ...rest
}: AuthFieldProps) {
  const fieldId = id ?? `field-${label.toLowerCase().replace(/\s+/g, "-")}`;
  const describedBy = error ? `${fieldId}-error` : hint ? `${fieldId}-hint` : undefined;

  return (
    <div className="afield">
      <label className="afield__label" htmlFor={fieldId}>
        {label}
      </label>
      <input
        id={fieldId}
        className={`afield__input ${error ? "afield__input--error" : ""} ${className}`}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy}
        {...rest}
      />
      {error ? (
        <p className="afield__error" id={`${fieldId}-error`}>
          {error}
        </p>
      ) : hint ? (
        <p className="afield__hint" id={`${fieldId}-hint`}>
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/** A password field with a show/hide toggle. */
export function PasswordField({
  label,
  error,
  hint,
  id,
  ...rest
}: Omit<AuthFieldProps, "type">) {
  const [visible, setVisible] = useState(false);
  const fieldId = id ?? `field-${label.toLowerCase().replace(/\s+/g, "-")}`;
  const describedBy = error ? `${fieldId}-error` : hint ? `${fieldId}-hint` : undefined;

  return (
    <div className="afield">
      <label className="afield__label" htmlFor={fieldId}>
        {label}
      </label>
      <div className="afield__wrap">
        <input
          id={fieldId}
          type={visible ? "text" : "password"}
          className={`afield__input afield__input--password ${error ? "afield__input--error" : ""}`}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          {...rest}
        />
        <button
          type="button"
          className="afield__toggle"
          onClick={() => setVisible((value) => !value)}
          // A typo in a masked field is the most common cause of a "wrong
          // password" that is not actually wrong.
          aria-label={visible ? "Hide password" : "Show password"}
          tabIndex={-1}
        >
          {visible ? <EyeOff /> : <Eye />}
        </button>
      </div>
      {error ? (
        <p className="afield__error" id={`${fieldId}-error`}>
          {error}
        </p>
      ) : hint ? (
        <p className="afield__hint" id={`${fieldId}-hint`}>
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/** The form-level error banner, for what the server rejected. */
export function AuthError({ children }: { children: ReactNode }) {
  return (
    <div className="aerror" role="alert">
      <IconAlert size={16} />
      <span>{children}</span>
    </div>
  );
}

/**
 * A password strength meter.
 *
 * Deliberately advisory, not a gate: the only rule enforced is the server's
 * minimum length. Blocking on a client-side entropy guess trains people to add
 * "!1" to the end, which helps nobody.
 */
export function PasswordStrength({ password }: { password: string }) {
  if (!password) return null;

  const checks = [
    password.length >= 8,
    password.length >= 12,
    /[a-z]/.test(password) && /[A-Z]/.test(password),
    /\d/.test(password) || /[^\w\s]/.test(password),
  ];
  const score = checks.filter(Boolean).length;
  const labels = ["Too short", "Weak", "Fair", "Good", "Strong"];
  const tones = ["critical", "critical", "warning", "good", "good"];

  return (
    <div className="strength">
      <div className="strength__track">
        {[0, 1, 2, 3].map((index) => (
          <span
            key={index}
            className={`strength__bar ${index < score ? `strength__bar--${tones[score]}` : ""}`}
          />
        ))}
      </div>
      <span className={`strength__label strength__label--${tones[score]}`}>
        {labels[score]}
      </span>
    </div>
  );
}

function Eye() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function EyeOff() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10.6 5.2A9.9 9.9 0 0 1 12 5c6.4 0 10 7 10 7a17.7 17.7 0 0 1-2.4 3.4M6.6 6.6A17.6 17.6 0 0 0 2 12s3.6 7 10 7a9.8 9.8 0 0 0 4.2-.9" />
      <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2M2 2l20 20" />
    </svg>
  );
}
