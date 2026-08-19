/**
 * Holds the signed-in user for the whole app.
 *
 * On start-up it calls `/auth/me` with any stored token rather than trusting
 * the token's own claims. That round trip is what makes a deleted or demoted
 * account stop working immediately instead of when its token happens to expire.
 */

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { auth as authApi, setUnauthorizedHandler, tokenStore } from "../../api/client";
import type { AuthUser } from "../../api/types";
import { AuthContext, type AuthContextValue } from "../../hooks/auth-context";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const signOut = useCallback(() => {
    tokenStore.clear();
    setUser(null);
  }, []);

  // Any 401 from anywhere in the app means the session is over.
  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));
    return () => setUnauthorizedHandler(null);
  }, []);

  // Restore the session on start-up, if a token is stored.
  useEffect(() => {
    const token = tokenStore.get();
    if (!token) {
      setLoading(false);
      return;
    }

    let cancelled = false;
    authApi
      .me()
      .then((restored) => {
        if (!cancelled) setUser(restored);
      })
      .catch(() => {
        // A stale or revoked token: drop it and stay signed out.
        if (!cancelled) {
          tokenStore.clear();
          setUser(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const result = await authApi.login({ email, password });
    tokenStore.set(result.access_token);
    setUser(result.user);
    return result.user;
  }, []);

  const signUp = useCallback(async (email: string, name: string, password: string) => {
    const result = await authApi.signup({ email, name, password });
    tokenStore.set(result.access_token);
    setUser(result.user);
    return result.user;
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      signIn,
      signUp,
      signOut,
      isAuthenticated: user !== null,
      isAdmin: user?.role === "admin",
    }),
    [user, loading, signIn, signUp, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
