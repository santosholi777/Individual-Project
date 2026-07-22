/**
 * The auth context and its hook.
 *
 * Separate from the provider component so React Fast Refresh keeps working —
 * a module that exports both a hook and a component gets a full reload on every
 * edit instead of a hot swap.
 */

import { createContext, useContext } from "react";
import type { AuthUser } from "../api/types";

export interface AuthContextValue {
  /** The signed-in account, or null when signed out. */
  user: AuthUser | null;
  /** True until the stored token has been checked on start-up. */
  loading: boolean;
  signIn: (email: string, password: string) => Promise<AuthUser>;
  signUp: (email: string, name: string, password: string) => Promise<AuthUser>;
  signOut: () => void;
  isAuthenticated: boolean;
  isAdmin: boolean;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

/** Access the signed-in user. Must be used below <AuthProvider>. */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside an <AuthProvider>");
  return context;
}
