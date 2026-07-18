/**
 * The single place this app talks to the AI service.
 *
 * Every call funnels through `request()`, so the service's structured error body
 * ({ error_code, message, details }) becomes a typed `ApiError` exactly once,
 * and no component ever touches `fetch` or a raw URL.
 */

import type {
  ApiErrorBody,
  AttendanceListResponse,
  AttendanceOutcome,
  AttendanceQuery,
  AttendanceSummaryResponse,
  AuthUser,
  DeleteResponse,
  ForgotPasswordResponse,
  HealthResponse,
  LoginPayload,
  MarkAttendancePayload,
  MessageResponse,
  RecognizePayload,
  RecognizeResponse,
  RegisterPayload,
  RegisterResponse,
  ResetPasswordPayload,
  SignupPayload,
  Student,
  StudentListResponse,
  TokenResponse,
  VerifyResetTokenResponse,
} from "./types";

/** Base URL of the Python AI service. Override with VITE_API_URL. */
export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const TOKEN_KEY = "dva-token";

/**
 * The access token, held in localStorage.
 *
 * localStorage is readable by any script on the page, so a successful XSS could
 * steal the token. The stronger option is an httpOnly cookie, which JavaScript
 * cannot read — but that needs cookie/CSRF handling on the API, which this
 * service does not implement. This is the standard bearer-token approach and is
 * appropriate for a local deployment; it is a documented limitation, not an
 * oversight.
 */
export const tokenStore = {
  get(): string | null {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null; // Private browsing can throw on access.
    }
  },
  set(token: string): void {
    try {
      localStorage.setItem(TOKEN_KEY, token);
    } catch {
      /* Non-fatal: the session simply will not survive a reload. */
    }
  },
  clear(): void {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* Nothing to do. */
    }
  },
};

/** Called when the API rejects our token, so the app can sign out cleanly. */
let onUnauthorized: (() => void) | null = null;

/**
 * Register a callback for expired/invalid sessions.
 *
 * A token can expire mid-session, so every request is a chance to discover we
 * are no longer signed in. Wiring this centrally means each page does not have
 * to handle it.
 */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler;
}

/**
 * An error returned by the AI service, carrying its stable `error_code` so
 * callers can branch on the cause rather than parse the message text.
 */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown>;

  constructor(body: ApiErrorBody, status: number) {
    super(body.message);
    this.name = "ApiError";
    this.code = body.error_code;
    this.status = status;
    this.details = body.details ?? {};
  }

  /** True when the AI service is unreachable rather than refusing the request. */
  get isOffline(): boolean {
    return this.code === "network_error";
  }
}

/** Shape of FastAPI's own validation failures, which differ from ours. */
interface FastApiValidationBody {
  detail?: { loc?: (string | number)[]; msg?: string }[] | string;
}

async function request<T>(
  path: string,
  init?: RequestInit,
  options: { anonymous?: boolean } = {},
): Promise<T> {
  let response: Response;

  const token = options.anonymous ? null : tokenStore.get();
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch (cause) {
    // fetch only rejects for network-level failures, which for this app almost
    // always means the Python service is not running.
    throw new ApiError(
      {
        error_code: "network_error",
        message:
          "Cannot reach the AI service. Make sure it is running: `uvicorn app:app` in the ai-service folder.",
        details: { cause: String(cause), base: API_BASE },
      },
      0,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const body: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const error = new ApiError(toErrorBody(body, response.status), response.status);
    // A 401 on any request means this session is over — sign out once, centrally,
    // rather than leaving each page to discover it. The login endpoints are
    // exempt: a wrong password there is a form error, not an expired session.
    if (response.status === 401 && !options.anonymous) {
      tokenStore.clear();
      onUnauthorized?.();
    }
    throw error;
  }

  return body as T;
}

/** Normalise both our error shape and FastAPI's validation shape into one. */
function toErrorBody(body: unknown, status: number): ApiErrorBody {
  if (body && typeof body === "object" && "error_code" in body) {
    return body as ApiErrorBody;
  }

  const validation = body as FastApiValidationBody | null;
  if (validation?.detail) {
    const message =
      typeof validation.detail === "string"
        ? validation.detail
        : validation.detail
            .map((item) => {
              const field = item.loc?.filter((part) => part !== "body").join(".");
              return field ? `${field}: ${item.msg}` : item.msg;
            })
            .join("; ");
    return {
      error_code: "validation_error",
      message: message || "The request was rejected as invalid.",
      details: {},
    };
  }

  return {
    error_code: "unknown_error",
    message: `The AI service returned an unexpected ${status} response.`,
    details: {},
  };
}

function jsonPost(payload: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

function toQueryString(params: Record<string, string | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value) search.set(key, value);
  }
  const query = search.toString();
  return query ? `?${query}` : "";
}

export const auth = {
  /** Create an account. The first account created becomes an admin. */
  signup: (payload: SignupPayload): Promise<TokenResponse> =>
    request("/auth/signup", jsonPost(payload), { anonymous: true }),

  /** Exchange an email and password for an access token. */
  login: (payload: LoginPayload): Promise<TokenResponse> =>
    request("/auth/login", jsonPost(payload), { anonymous: true }),

  /** Resolve the token we hold back to an account, or fail if it is stale. */
  me: (): Promise<AuthUser> => request("/auth/me"),

  /** Request a password-reset link. */
  forgotPassword: (email: string): Promise<ForgotPasswordResponse> =>
    request("/auth/forgot-password", jsonPost({ email }), { anonymous: true }),

  /** Check whether a reset link is still usable, before showing the form. */
  verifyResetToken: (token: string): Promise<VerifyResetTokenResponse> =>
    request(
      `/auth/reset-password/verify${toQueryString({ token })}`,
      undefined,
      { anonymous: true },
    ),

  /** Set a new password using a reset token. */
  resetPassword: (payload: ResetPasswordPayload): Promise<MessageResponse> =>
    request("/auth/reset-password", jsonPost(payload), { anonymous: true }),
};

export const api = {
  /** Liveness, model and gallery-index status. Public. */
  health: (): Promise<HealthResponse> => request("/health", undefined, { anonymous: true }),

  /** Enrol a student from base64 webcam frames. */
  register: (payload: RegisterPayload): Promise<RegisterResponse> =>
    request("/register/base64", jsonPost(payload)),

  /** Recognise every face in a base64 frame, optionally marking attendance. */
  recognize: (payload: RecognizePayload): Promise<RecognizeResponse> =>
    request("/recognize/base64", jsonPost(payload)),

  /** List every registered student. */
  listStudents: (): Promise<StudentListResponse> => request("/students"),

  /** Fetch one student by id. */
  getStudent: (id: string): Promise<Student> =>
    request(`/students/${encodeURIComponent(id)}`),

  /** Delete a student and every face embedding held for them. */
  deleteStudent: (id: string): Promise<DeleteResponse> =>
    request(`/students/${encodeURIComponent(id)}`, { method: "DELETE" }),

  /** Mark attendance manually, without an image. */
  markAttendance: (payload: MarkAttendancePayload): Promise<AttendanceOutcome> =>
    request("/attendance", jsonPost(payload)),

  /** Query attendance records with optional filters. */
  listAttendance: (query: AttendanceQuery = {}): Promise<AttendanceListResponse> =>
    request(`/attendance${toQueryString(query as Record<string, string | undefined>)}`),

  /** Present/absent totals for one day. */
  attendanceSummary: (date?: string): Promise<AttendanceSummaryResponse> =>
    request(`/attendance/summary${toQueryString({ date })}`),
};
