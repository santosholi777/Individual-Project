/**
 * Sign in.
 *
 * The API deliberately returns the same error for a wrong password and an
 * unknown address, and this page shows it verbatim rather than trying to be
 * more helpful — being more helpful here means telling a stranger which
 * addresses have accounts.
 */

import { useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ApiError } from "../../api/client";
import { Button } from "../../components/ui/Button";
import { useToast } from "../../components/ui/toast-context";
import { useAuth } from "../../hooks/auth-context";
import { AuthError, AuthField, PasswordField } from "./AuthForm";
import { AuthLayout } from "./AuthLayout";

interface LocationState {
  from?: string;
}

export function Login() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const toast = useToast();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Where the guard bounced them from, so they land back there after signing in.
  const destination = (location.state as LocationState | null)?.from ?? "/";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const user = await signIn(email.trim(), password);
      toast.success(`Welcome back, ${user.name.split(" ")[0]}`);
      navigate(destination, { replace: true });
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
      title="Sign in"
      subtitle="Welcome back. Sign in to manage attendance."
      footer={
        <>
          Don&apos;t have an account? <Link to="/signup">Create one</Link>
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
        />

        <PasswordField
          label="Password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          placeholder="Your password"
          autoComplete="current-password"
          required
          disabled={loading}
        />

        <div className="aform__row">
          <span />
          <Link to="/forgot-password" className="aform__link">
            Forgot password?
          </Link>
        </div>

        <Button
          type="submit"
          variant="primary"
          size="lg"
          fullWidth
          loading={loading}
          disabled={!email.trim() || !password}
        >
          {loading ? "Signing in…" : "Sign in"}
        </Button>
      </form>
    </AuthLayout>
  );
}
