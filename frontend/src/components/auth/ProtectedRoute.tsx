/**
 * Route guards.
 *
 * These are a **convenience, not a security boundary**. Anyone can edit the
 * JavaScript running in their own browser, so the guard that actually matters is
 * the one on the API. This exists so a signed-out user sees the login page
 * instead of a screen full of failed requests.
 */

import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../../hooks/auth-context";
import { LoadingState } from "../ui/States";

/** Requires any signed-in account. */
export function ProtectedRoute() {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();

  // Wait for the start-up /auth/me check: redirecting first would bounce a
  // signed-in user to the login page on every refresh.
  if (loading) return <LoadingState label="Restoring your session…" />;

  if (!isAuthenticated) {
    // Remember where they were headed so login can send them back.
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  return <Outlet />;
}

/** Requires the admin role. */
export function AdminRoute() {
  const { isAdmin, loading } = useAuth();

  if (loading) return <LoadingState label="Checking permissions…" />;
  if (!isAdmin) return <Navigate to="/" replace />;
  return <Outlet />;
}

/**
 * For the auth pages themselves: bounces an already-signed-in user to the app.
 *
 * Without this, the browser's back button lands a signed-in user on a login
 * form, which is confusing.
 */
export function PublicOnlyRoute() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) return <LoadingState label="Loading…" />;
  if (isAuthenticated) return <Navigate to="/" replace />;
  return <Outlet />;
}
