/**
 * Route table.
 *
 * Two zones: the auth pages, which bounce signed-in users away, and everything
 * under the AppShell, which requires an account.
 *
 * These guards are for usability, not security. The real enforcement is on the
 * API — a browser guard only decides what gets rendered, and anyone can edit
 * their own JavaScript.
 */

import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./components/auth/AuthProvider";
import { ProtectedRoute, PublicOnlyRoute } from "./components/auth/ProtectedRoute";
import { AppShell } from "./components/layout/AppShell";
import { ToastProvider } from "./components/ui/Toast";
import { AttendanceLog } from "./pages/AttendanceLog";
import { Dashboard } from "./pages/Dashboard";
import { Kiosk } from "./pages/Kiosk";
import { Register } from "./pages/Register";
import { Students } from "./pages/Students";
import { ForgotPassword } from "./pages/auth/ForgotPassword";
import { Login } from "./pages/auth/Login";
import { ResetPassword } from "./pages/auth/ResetPassword";
import { Signup } from "./pages/auth/Signup";

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            {/* Signed-out only */}
            <Route element={<PublicOnlyRoute />}>
              <Route path="/login" element={<Login />} />
              <Route path="/signup" element={<Signup />} />
              <Route path="/forgot-password" element={<ForgotPassword />} />
            </Route>

            {/* Reachable while signed in too: a reset link arrives by email and
                may be opened in any session. */}
            <Route path="/reset-password" element={<ResetPassword />} />

            {/* Signed-in only. Students is visible to everyone, but deleting a
                student is admin-only — enforced by the API and mirrored in the
                page's UI. */}
            <Route element={<ProtectedRoute />}>
              <Route element={<AppShell />}>
                <Route index element={<Dashboard />} />
                <Route path="kiosk" element={<Kiosk />} />
                <Route path="register" element={<Register />} />
                <Route path="students" element={<Students />} />
                <Route path="attendance" element={<AttendanceLog />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ToastProvider>
  );
}
