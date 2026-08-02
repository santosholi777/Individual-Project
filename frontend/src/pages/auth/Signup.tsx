/**
 * Create an account.
 *
 * The confirm-password check is done here rather than server-side: it exists to
 * catch a typo, and a round trip to learn you mistyped is a worse experience for
 * no security gain. Everything that actually matters — length, uniqueness,
 * hashing — is enforced by the API.
 */

import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError } from "../../api/client";
import { Button } from "../../components/ui/Button";
import { useToast } from "../../components/ui/toast-context";
import { useAuth } from "../../hooks/auth-context";
import { AuthError, AuthField, PasswordField, PasswordStrength } from "./AuthForm";
import { AuthLayout } from "./AuthLayout";

/** Must match the server's DVA_PASSWORD_MIN_LENGTH. */
const MIN_PASSWORD_LENGTH = 8;

export function Signup() {
  const { signUp } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const mismatch = confirm.length > 0 && password !== confirm;
  const tooShort = password.length > 0 && password.length < MIN_PASSWORD_LENGTH;
  const canSubmit =
    name.trim() && email.trim() && password.length >= MIN_PASSWORD_LENGTH && password === confirm;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);

    if (password !== confirm) {
      setError("The two passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      const user = await signUp(email.trim(), name.trim(), password);
      toast.success(
        `Account created for ${user.name}`,
        user.role === "admin"
          ? "You are the first account, so you have administrator access."
          : "You are signed in.",
      );
      navigate("/", { replace: true });
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

  return (
    <AuthLayout
      title="Create an account"
      subtitle="Set up access to the attendance system."
      footer={
        <>
          Already have an account? <Link to="/login">Sign in</Link>
        </>
      }
    >
      <form className="aform" onSubmit={submit} noValidate>
        {error && <AuthError>{error}</AuthError>}

        <AuthField
          label="Full name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Dr. Meera Krishnan"
          autoComplete="name"
          required
          autoFocus
          disabled={loading}
        />

        <AuthField
          label="Email"
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@college.edu"
          autoComplete="email"
          required
          disabled={loading}
          hint="This is what you will sign in with."
        />

        <div>
          <PasswordField
            label="Password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="At least 8 characters"
            autoComplete="new-password"
            required
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
          label="Confirm password"
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
          {loading ? "Creating account…" : "Create account"}
        </Button>

        <p className="afield__hint" style={{ textAlign: "center" }}>
          The first account created becomes the administrator.
        </p>
      </form>
    </AuthLayout>
  );
}
