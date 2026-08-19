/**
 * Small data-fetching hooks.
 *
 * Deliberately hand-rolled rather than pulling in a data library: this app has a
 * handful of endpoints and no cache-invalidation problem worth a dependency.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export interface QueryState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  /** Re-run the fetch, e.g. after a mutation or on a manual refresh. */
  refetch: () => void;
}

/**
 * Run an async API call on mount and whenever `deps` change.
 *
 * @param fetcher The API call to run.
 * @param deps Values that should trigger a refetch when they change.
 */
export function useQuery<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
): QueryState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  // Kept in a ref so changing the fetcher identity every render (the common
  // case with inline arrow functions) does not cause an infinite refetch loop.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    fetcherRef
      .current()
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(
          cause instanceof ApiError
            ? cause
            : new ApiError(
                {
                  error_code: "unknown_error",
                  message: String(cause),
                  details: {},
                },
                0,
              ),
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const refetch = useCallback(() => setNonce((value) => value + 1), []);

  return { data, error, loading, refetch };
}

export interface MutationState<TArgs, TResult> {
  run: (args: TArgs) => Promise<TResult>;
  loading: boolean;
  error: ApiError | null;
  reset: () => void;
}

/**
 * Wrap a write call so components get `loading`/`error` without try/catch noise.
 *
 * `run` still rejects on failure, so callers that need to react to the outcome
 * can await it; the state is a convenience for rendering, not a replacement.
 */
export function useMutation<TArgs, TResult>(
  action: (args: TArgs) => Promise<TResult>,
): MutationState<TArgs, TResult> {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const actionRef = useRef(action);
  actionRef.current = action;

  const run = useCallback(async (args: TArgs): Promise<TResult> => {
    setLoading(true);
    setError(null);
    try {
      return await actionRef.current(args);
    } catch (cause) {
      const apiError =
        cause instanceof ApiError
          ? cause
          : new ApiError(
              { error_code: "unknown_error", message: String(cause), details: {} },
              0,
            );
      setError(apiError);
      throw apiError;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => setError(null), []);

  return { run, loading, error, reset };
}
