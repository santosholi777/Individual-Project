/**
 * The split-screen frame shared by all four auth pages.
 *
 * The left panel is a hand-built background: a layered gradient, a faint grid,
 * and an abstract face-scan motif drawn as inline SVG. It is deliberately not a
 * photograph — a vector stays sharp at any size, adds nothing to the bundle,
 * needs no licensing, and cannot fail to load. To use a real photo instead, drop
 * one in and set it as the `.auth__art` background-image; the layout is
 * unchanged.
 *
 * On narrow screens the art collapses to a compact header so the form still
 * gets the whole viewport.
 */

import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import "./AuthLayout.css";

export interface AuthLayoutProps {
  title: string;
  subtitle: string;
  children: ReactNode;
  /** Optional line under the form, e.g. a link to the other page. */
  footer?: ReactNode;
}

/** The abstract face-scan graphic. Decorative, so hidden from screen readers. */
function ScanArt() {
  return (
    <svg
      className="auth__art-svg"
      viewBox="0 0 400 400"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <linearGradient id="scan-stroke" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="rgba(255,255,255,0.85)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0.25)" />
        </linearGradient>
        <linearGradient id="scan-sweep" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="rgba(120,190,255,0)" />
          <stop offset="50%" stopColor="rgba(150,205,255,0.9)" />
          <stop offset="100%" stopColor="rgba(120,190,255,0)" />
        </linearGradient>
      </defs>

      {/* Detection bracket — the visual language of the kiosk's face box. */}
      <g stroke="url(#scan-stroke)" strokeWidth="2.5" strokeLinecap="round">
        <path d="M96 60H72a12 12 0 0 0-12 12v24" />
        <path d="M304 60h24a12 12 0 0 1 12 12v24" />
        <path d="M340 304v24a12 12 0 0 1-12 12h-24" />
        <path d="M60 304v24a12 12 0 0 0 12 12h24" />
      </g>

      {/* Abstract face: head, eyes, nose, mouth — the five ArcFace landmarks. */}
      <g opacity="0.9">
        <ellipse
          cx="200"
          cy="196"
          rx="74"
          ry="92"
          stroke="rgba(255,255,255,0.5)"
          strokeWidth="2"
        />
        <circle cx="172" cy="178" r="5.5" fill="rgba(150,205,255,0.95)" />
        <circle cx="228" cy="178" r="5.5" fill="rgba(150,205,255,0.95)" />
        <circle cx="200" cy="212" r="5.5" fill="rgba(150,205,255,0.95)" />
        <circle cx="178" cy="242" r="5.5" fill="rgba(150,205,255,0.95)" />
        <circle cx="222" cy="242" r="5.5" fill="rgba(150,205,255,0.95)" />
        {/* The landmark mesh the alignment step actually uses. */}
        <g stroke="rgba(150,205,255,0.35)" strokeWidth="1">
          <path d="M172 178 228 178 200 212 178 242 222 242 200 212M172 178 200 212M228 178 222 242M172 178 178 242" />
        </g>
      </g>

      {/* The sweep line, echoing the kiosk's scanning state. */}
      <rect className="auth__scanline" x="60" y="0" width="280" height="2.5" fill="url(#scan-sweep)" />
    </svg>
  );
}

export function AuthLayout({ title, subtitle, children, footer }: AuthLayoutProps) {
  return (
    <div className="auth">
      <aside className="auth__art">
        <div className="auth__art-grid" aria-hidden="true" />
        <div className="auth__art-glow" aria-hidden="true" />

        <div className="auth__brand">
          <span className="auth__brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" />
              <circle cx="12" cy="10.5" r="2.6" />
              <path d="M7.5 17.5a4.8 4.8 0 0 1 9 0" />
            </svg>
          </span>
          <span className="auth__brand-name">DeepVisionAttend</span>
        </div>

        <ScanArt />

        <div className="auth__art-copy">
          <h2 className="auth__art-title">Attendance, without the roll call.</h2>
          <p className="auth__art-text">
            Deep learning face recognition that marks students present in
            milliseconds — accurately, and without ever storing a photograph.
          </p>
          <ul className="auth__art-points">
            <li>Pre-trained ArcFace recognition</li>
            <li>No photos stored — only maths</li>
            <li>Proxy attendance made impossible</li>
          </ul>
        </div>
      </aside>

      <main className="auth__panel">
        <div className="auth__card">
          <Link to="/login" className="auth__mobile-brand" aria-label="DeepVisionAttend">
            <span className="auth__brand-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" />
                <circle cx="12" cy="10.5" r="2.6" />
                <path d="M7.5 17.5a4.8 4.8 0 0 1 9 0" />
              </svg>
            </span>
            <span className="auth__brand-name">DeepVisionAttend</span>
          </Link>

          <header className="auth__header">
            <h1 className="auth__title">{title}</h1>
            <p className="auth__subtitle">{subtitle}</p>
          </header>

          {children}

          {footer && <div className="auth__footer">{footer}</div>}
        </div>
      </main>
    </div>
  );
}
