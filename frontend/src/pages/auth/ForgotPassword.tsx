/**
 * Request a password-reset link.
 *
 * The success screen says "if an account exists" rather than confirming one
 * does — the API refuses to reveal which addresses are registered, and the UI
 * must not undo that by showing a different screen for a hit and a miss.
 *
 * In development the API returns the link, so it is shown on screen (clearly
 * flagged as a dev shortcut). In production that field is null and this page
 * simply tells the user to check their email.
 */

import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { ApiError, auth as authApi } from "../../api/client";
import { Button } from "../../components/ui/Button";
import { IconAlert, IconCheckCircle } from "../../components/ui/icons";
import { AuthError, AuthField } from "./AuthForm";
import { AuthLayout } from "./AuthLayout";
import "./ForgotPassword.css";

export function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [devLink, setDevLink] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await authApi.forgotPassword(email.trim());
      setDevLink(response.reset_link);
      setSent(true);
    } catch (cause) {
      if (cause instanceof ApiError) {
        setError(
          cause.isOffline
            ? "Cannot reach the server. Make sure the AI service is running."
            : cause.message,
        );
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  if (sent) {
    return (
      <AuthLayout
        title="Check your email"
        subtitle="If an account exists for that address, we've sent it a link to reset the password."
        footer={
          <>
            Remembered it? <Link to="/login">Back to sign in</Link>
          </>
        }
      >
        <div className="fp">
          <div className="fp__sent">
            <span className="fp__sent-icon">
              <IconCheckCircle size={20} />
            </span>
            <div>
              <p className="fp__sent-title">Reset link sent to {email}</p>
              <p className="fp__sent-text">
                The link expires in 30 minutes and can only be used once.
              </p>
            </div>
          </div>

          {devLink && (
            <div className="fp__dev">
              <div className="fp__dev-head">
                <IconAlert size={15} />
                <span>Development mode — no email was actually sent</span>
              </div>
              <p className="fp__dev-text">
                The server is configured with <code>DVA_EXPOSE_RESET_LINK=true</code>,
                so it returned the link instead of emailing it. Turn that off
                before any real deployment — it lets anyone reset any account.
              </p>
              <Link to={devLink.replace(window.location.origin, "")} className="fp__dev-link">
                Open the reset link →
              </Link>
            </div>
          )}

          <Button fullWidth onClick={() => setSent(false)}>
            Use a different email
          </Button>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Forgot password?"
      subtitle="Enter your email and we'll send you a link to set a new one."
      footer={
        <>
          Remembered it? <Link to="/login">Back to sign in</Link>
        </>
      }
    >
      <form className="aform" onSubmit={submit} noValidate>
        {error && <AuthError>{error}</AuthError>}

        <AuthField
          label="Email"
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@college.edu"
          autoComplete="email"
          required
          autoFocus
          disabled={loading}
          hint="We'll send a reset link if this address has an account."
        />

        <Button
          type="submit"
          variant="primary"
          size="lg"
          fullWidth
          loading={loading}
          disabled={!email.trim()}
        >
          {loading ? "Sending…" : "Send reset link"}
        </Button>
      </form>
    </AuthLayout>
  );
}
