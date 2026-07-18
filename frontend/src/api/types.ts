/**
 * TypeScript mirror of the AI service's HTTP contract.
 *
 * These types are hand-kept in sync with `ai-service/schemas.py`. If an endpoint
 * changes shape there, it changes here — that pairing is the contract.
 */

/** The body every 4xx/5xx from the AI service returns. */
export interface ApiErrorBody {
  error_code: string;
  message: string;
  details: Record<string, unknown>;
}

export interface Student {
  student_id: string;
  name: string;
  embedding_count: number;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface StudentListResponse {
  count: number;
  students: Student[];
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface MatchCandidate {
  student_id: string;
  name: string;
  similarity: number;
}

export interface RecognitionResult {
  recognized: boolean;
  student_id: string | null;
  name: string | null;
  /** Cosine similarity of the best match, 0–1. */
  confidence: number;
  /** Gap to the runner-up; a small value means an ambiguous match. */
  margin: number;
  bbox: BoundingBox;
  det_score: number;
  candidates: MatchCandidate[];
}

export type AttendanceStatus = "marked" | "duplicate" | "rejected";

export interface AttendanceRecord {
  student_id: string;
  name: string;
  timestamp: string;
  date: string;
  confidence: number;
  session: string;
  source: "auto" | "manual";
}

export interface AttendanceOutcome {
  status: AttendanceStatus;
  reason: string | null;
  record: AttendanceRecord | null;
}

export interface RecognizeResponse {
  success: boolean;
  faces_detected: number;
  recognized_count: number;
  elapsed_ms: number;
  results: RecognitionResult[];
  /** Populated only when attendance marking was requested. */
  attendance: AttendanceOutcome[];
}

export interface RegisterResponse {
  success: boolean;
  student: Student;
  accepted_images: number;
  rejected_images: number;
  rejections: string[];
  total_embeddings: number;
}

export interface AttendanceListResponse {
  count: number;
  records: AttendanceRecord[];
}

export interface AttendanceSummaryResponse {
  date: string;
  total_students: number;
  present: number;
  absent: number;
  attendance_rate: number;
  records: AttendanceRecord[];
  absentees: { student_id: string; name: string }[];
}

export interface HealthResponse {
  status: "ok" | "degraded";
  service: string;
  version: string;
  models_ready: boolean;
  model_info: Record<string, unknown>;
  index: {
    indexed_embeddings: number;
    indexed_students: number;
    recognition_threshold: number;
  };
  storage: Record<string, unknown>;
  timestamp: string;
}

export interface DeleteResponse {
  success: boolean;
  message: string;
}

export interface RegisterPayload {
  student_id: string;
  name: string;
  images: string[];
  overwrite?: boolean;
  metadata?: Record<string, unknown>;
}

export interface RecognizePayload {
  image: string;
  mark_attendance?: boolean;
  session?: string | null;
  max_faces?: number | null;
}

export interface AttendanceQuery {
  student_id?: string;
  date_from?: string;
  date_to?: string;
  session?: string;
}

export interface MarkAttendancePayload {
  student_id: string;
  confidence?: number;
  session?: string | null;
  source?: "auto" | "manual";
}

// ======================================================================
// Authentication
// ======================================================================

export type UserRole = "admin" | "lecturer";

export interface AuthUser {
  user_id: string;
  email: string;
  name: string;
  role: UserRole;
  created_at: string;
  last_login_at: string | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  /** Token lifetime in seconds. */
  expires_in: number;
  user: AuthUser;
}

export interface SignupPayload {
  email: string;
  name: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface ForgotPasswordResponse {
  success: boolean;
  message: string;
  /** Development only — null once DVA_EXPOSE_RESET_LINK is off. */
  reset_token: string | null;
  /** Development only — in production this is emailed instead. */
  reset_link: string | null;
}

export interface ResetPasswordPayload {
  token: string;
  password: string;
}

export interface MessageResponse {
  success: boolean;
  message: string;
}

export interface VerifyResetTokenResponse {
  valid: boolean;
}
