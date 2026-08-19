/**
 * Set a new password from a reset link.
 *
 * The token is checked with the server *before* the form is shown. Letting
 * someone type a new password twice only to be told the link expired is a
 * needlessly cruel way to deliver that news.
 */

import { useEffect, useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ApiError, auth as authApi } from "../../api/client";
import { Button } from "../../components/ui/Button";
import { IconAlert, IconCheckCircle } from "../../components/ui/icons";
import { Spinner } from "../../components/ui/States";
import { useToast } from "../../components/ui/toast-context";
import { AuthError, PasswordField, PasswordStrength } from "./AuthForm";
import { AuthLayout } from "./AuthLayout";
import "./ForgotPassword.css";

const MIN_PASSWORD_LENGTH = 8;

type TokenState = "checking" | "valid" | "invalid" | "missing";

export function ResetPassword() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const toast = useToast();
  const token = params.get("token");

  const [tokenState, setTokenState] = useState<TokenState>("checking");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!token) {
      setTokenState("missing");
      return;
    }

    let cancelled = false;
    authApi
      .verifyResetToken(token)
      .then((result) => {
        if (!cancelled) setTokenState(result.valid ? "valid" : "invalid");
      })
      .catch(() => {
        if (!cancelled) setTokenState("invalid");
      });

    return () => {
      cancelled = true;
    };
  }, [token]);

  const mismatch = confirm.length > 0 && password !== confirm;
  const tooShort = password.length > 0 && password.length < MIN_PASSWORD_LENGTH;
  const canSubmit = password.length >= MIN_PASSWORD_LENGTH && password === confirm;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!token) return;

    setError(null);
    setLoading(true);
    try {
      await authApi.resetPassword({ token, password });
      setDone(true);
      toast.success("Password updated", "You can now sign in with your new password.");
      // Give the success screen a beat before redirecting.
      window.setTimeout(() => navigate("/login", { replace: true }), 2500);
    } catch (cause) {
      if (cause instanceof ApiError) {
        setError(cause.message);
        if (cause.code === "invalid_reset_token") setTokenState("invalid");
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  if (tokenState === "checking") {
    return (
      <AuthLayout title="Checking your link" subtitle="One moment…">
        <div className="fp__checking">
          <Spinner size={24} />
        </div>
      </AuthLayout>
    );
  }

  if (tokenState === "missing" || tokenState === "invalid") {
    return (
      <AuthLayout
        title="This link doesn't work"
        subtitle={
          tokenState === "missing"
            ? "The link is missing its reset token. Copy the whole link from your email."
            : "This reset link is invalid, has already been used, or has expired."
        }
        footer={
          <>
            Remembered it? <Link to="/login">Back to sign in</Link>
          </>
        }
      >
        <div className="fp">
          <div className="fp__invalid">
            <span className="fp__invalid-icon">
              <IconAlert size={20} />
            </span>
            <p>
              Reset links expire after 30 minutes and can only be used once.
              Request a fresh one and use it straight away.
            </p>
          </div>
          <Link to="/forgot-password" className="unstyled-link">
            <Button variant="primary" size="lg" fullWidth>
              Request a new link
            </Button>
          </Link>
        </div>
      </AuthLayout>
    );
  }

  if (done) {
    return (
      <AuthLayout title="Password updated" subtitle="You're all set.">
        <div className="fp">
          <div className="fp__sent">
            <span className="fp__sent-icon">
              <IconCheckCircle size={20} />
            </span>
            <div>
              <p className="fp__sent-title">Your password has been changed</p>
              <p className="fp__sent-text">Taking you to the sign-in page…</p>
            </div>
          </div>
          <Link to="/login" className="unstyled-link">
            <Button variant="primary" size="lg" fullWidth>
              Sign in now
            </Button>
          </Link>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Set a new password"
      subtitle="Choose a new password for your account."
      footer={
        <>
          Changed your mind? <Link to="/login">Back to sign in</Link>
        </>
      }
    >
      <form className="aform" onSubmit={submit} noValidate>
        {error && <AuthError>{error}</AuthError>}

        <div>
          <PasswordField
            label="New password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="At least 8 characters"
            autoComplete="new-password"
            required
            autoFocus
            disabled={loading}
            error={tooShort ? `Use at least ${MIN_PASSWORD_LENGTH} characters.` : null}
          />
          {password && !tooShort && (
            <div style={{ marginTop: "var(--s2)" }}>
              <PasswordStrength password={password} />
            </div>
          )}
        </div>

        <PasswordField
          label="Confirm new password"
          value={confirm}
          onChange={(event) => setConfirm(event.target.value)}
          placeholder="Type it again"
          autoComplete="new-password"
          required
          disabled={loading}
          error={mismatch ? "The two passwords do not match." : null}
        />

        <Button
          type="submit"
          variant="primary"
          size="lg"
          fullWidth
          loading={loading}
          disabled={!canSubmit}
        >
          {loading ? "Updating…" : "Update password"}
        </Button>
      </form>
    </AuthLayout>
  );
}
