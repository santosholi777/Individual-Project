/**
 * The application frame: brand, navigation, service status and theme toggle.
 *
 * The sidebar carries a live health indicator. Every failure in this app traces
 * back to "is the Python service running?", so that question is answered
 * permanently on screen instead of through a stack of confusing errors.
 */

import { useEffect, useRef } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { useQuery } from "../../hooks/useApi";
import { useAuth } from "../../hooks/auth-context";
import { useTheme } from "../../hooks/useTheme";
import {
  IconDashboard,
  IconList,
  IconLogout,
  IconMoon,
  IconScan,
  IconSun,
  IconUserPlus,
  IconUsers,
} from "../ui/icons";
import "./AppShell.css";

const NAV = [
  { to: "/", label: "Dashboard", icon: IconDashboard, end: true },
  { to: "/kiosk", label: "Take Attendance", icon: IconScan, end: false },
  { to: "/register", label: "Register Student", icon: IconUserPlus, end: false },
  { to: "/students", label: "Students", icon: IconUsers, end: false },
  { to: "/attendance", label: "Attendance Log", icon: IconList, end: false },
];

/** Poll interval for the health indicator. Cheap endpoint, slow cadence. */
const HEALTH_POLL_MS = 20000;

function ServiceStatus() {
  const { data, error, refetch } = useQuery(() => api.health(), []);

  // Re-check on a timer so the dot reflects a service that died mid-session.
  useTicker(refetch, HEALTH_POLL_MS);

  const tone = error ? "offline" : data?.models_ready ? "online" : "degraded";
  const label = error
    ? "Service offline"
    : data?.models_ready
      ? "AI service online"
      : "Models loading…";

  return (
    <div className="status" title={error ? error.message : label}>
      <span className={`status__dot status__dot--${tone}`} aria-hidden="true" />
      <div className="status__text">
        <span className="status__label">{label}</span>
        {data && (
          <span className="status__meta">
            {data.index.indexed_students} enrolled · v{data.version}
          </span>
        )}
        {error && <span className="status__meta">Start the Python service</span>}
      </div>
    </div>
  );
}

/** Run `callback` every `ms`, without re-subscribing on every render. */
function useTicker(callback: () => void, ms: number) {
  const saved = useRef(callback);
  saved.current = callback;

  useEffect(() => {
    const id = window.setInterval(() => saved.current(), ms);
    return () => window.clearInterval(id);
  }, [ms]);
}

/** The signed-in account, with sign-out. */
function UserMenu() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  if (!user) return null;

  const handleSignOut = () => {
    signOut();
    navigate("/login", { replace: true });
  };

  return (
    <div className="user">
      <span className="user__avatar" aria-hidden="true">
        {user.name.charAt(0).toUpperCase()}
      </span>
      <div className="user__text">
        <span className="user__name" title={user.name}>
          {user.name}
        </span>
        <span className="user__role">
          {user.role === "admin" ? "Administrator" : "Lecturer"}
        </span>
      </div>
      <button
        className="user__signout"
        onClick={handleSignOut}
        aria-label="Sign out"
        title="Sign out"
      >
        <IconLogout size={16} />
      </button>
    </div>
  );
}

export function AppShell() {
  const { resolved, toggle } = useTheme();

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar__brand">
          <div className="brand__mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" />
              <circle cx="12" cy="10.5" r="2.6" />
              <path d="M7.5 17.5a4.8 4.8 0 0 1 9 0" />
            </svg>
          </div>
          <div className="brand__text">
            <span className="brand__name">DeepVisionAttend</span>
            <span className="brand__tag">Smart Attendance</span>
          </div>
        </div>

        <nav className="sidebar__nav" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `nav__item ${isActive ? "nav__item--active" : ""}`
              }
            >
              <Icon size={17} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__footer">
          <ServiceStatus />
          <button
            className="theme-toggle"
            onClick={toggle}
            aria-label={`Switch to ${resolved === "dark" ? "light" : "dark"} theme`}
          >
            {resolved === "dark" ? <IconSun size={16} /> : <IconMoon size={16} />}
            <span>{resolved === "dark" ? "Light mode" : "Dark mode"}</span>
          </button>
          <UserMenu />
        </div>
      </aside>

      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
