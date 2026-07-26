/**
 * The toast context and its hook.
 *
 * Kept apart from `Toast.tsx` because React Fast Refresh only works on modules
 * that export components exclusively — mixing the hook in with the provider
 * makes every edit to a toast do a full page reload instead of a hot swap.
 */

import { createContext, useContext } from "react";

export type ToastTone = "success" | "error" | "info";

export interface ToastContextValue {
  success: (title: string, description?: string) => void;
  error: (title: string, description?: string) => void;
  info: (title: string, description?: string) => void;
}

export const ToastContext = createContext<ToastContextValue | null>(null);

/** Raise toasts from anywhere below <ToastProvider>. */
export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used inside a <ToastProvider>");
  return context;
}
